import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs'; //the underlying numerical engine that powers the detection algorithms.
import * as poseDetection from '@tensorflow-models/pose-detection'; //defines the network architecture and skeleton keypoints
import '@tensorflow/tfjs-backend-webgl'; // used for WebGL backend in TensorFlow.js

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

/**
 * Get the specified joint from the pose.
 * @param {Object} pose - The pose object.
 * @param {string} jointName - The name of the joint to retrieve.
 * @returns {Object|null} The joint keypoint or null if not found or low confidence.
 */
const getJoint = (pose, jointName) => {
  if (!pose) return null;
  const keypoint = pose.keypoints?.find((point) => point.name === jointName);
  if (!keypoint || keypoint.score < 0.3) return null;
  return keypoint;
};

/**
 * Calculate the angle formed by three points (a, b, c) with b as the vertex.
 * @param {Object} a - The first point.
 * @param {Object} b - The vertex point.
 * @param {Object} c - The third point.
 * @returns {number} The angle in degrees.
 */
const calculateAngle = (a, b, c) => {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const magnitude = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);

  // If the magnitude is too small, the points are almost collinear or coincident, return 180 degrees as a default.
  if (magnitude <= 0.0001) return 180;

  const cosine = clamp(dot / magnitude, -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
};

/**
 * Draw a line between two points on the canvas.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {Object} from - The starting point.
 * @param {Object} to - The ending point.
 * @param {string} color - The color of the line.
 * @param {number} width - The width of the line.
 */
const drawLine = (ctx, from, to, color = '#a78bfa', width = 4) => {
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
};

/**
 * Draw a joint (circle) on the canvas.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {Object} point - The point representing the joint.
 * @param {string} color - The color of the joint.
 * @param {number} radius - The radius of the joint.
 */
const drawJoint = (ctx, point, color = '#34d399', radius = 7) => {
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius + 2, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 2;
  ctx.stroke();
};

/**
 * Draw an arc representing the angle at point b formed by points a, b, and c.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {Object} a - The first point.
 * @param {Object} b - The vertex point.
 * @param {Object} c - The third point.
 * @param {number} angleDegrees - The angle in degrees.
 */
const drawAngleArc = (ctx, a, b, c, angleDegrees) => {
  const centerX = b.x;
  const centerY = b.y;
  const radius = 38;

  const vectorA = { x: a.x - centerX, y: a.y - centerY };
  const vectorC = { x: c.x - centerX, y: c.y - centerY };

  const startAngle = Math.atan2(vectorA.y, vectorA.x);
  const endAngle = Math.atan2(vectorC.y, vectorC.x);

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, startAngle, endAngle, false);
  ctx.strokeStyle = angleDegrees < 90 ? '#22c55e' : '#f97316';
  ctx.lineWidth = 5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius - 10, startAngle, endAngle, false);
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 2;
  ctx.stroke();
};

/**
 * The main App component.
 * @returns {JSX.Element} The rendered App component.
 */
function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const rafRef = useRef(0);

  const [status, setStatus] = useState('Starting camera...');
  const [angleValue, setAngleValue] = useState(0);
  const [feedback, setFeedback] = useState('Stand ready');
  const [fps, setFps] = useState(0);
  const [isReady, setIsReady] = useState(false);

  /**
   * UseEffect hook to initialize the camera and pose detection model.
   */
  useEffect(() => {
    let cancelled = false;

    const startCamera = async () => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error('Webcam not supported in this browser.');
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (!videoRef.current) return;

        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();

        if (!cancelled) {
          setStatus('Camera live');
          setIsReady(true);
          detectLoop();
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setStatus('Camera access denied. Please allow webcam access.');
        }
      }
    };

    /**
     * initialize the TensorFlow model and camera.
     */
    const init = async () => {
      try {
        await tf.ready();
        await tf.setBackend('webgl');

        detectorRef.current = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        });

        if (!cancelled) {
          setStatus('Model loaded');
          await startCamera();
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setStatus('Pose model failed to load. Please refresh and allow webcam access.');
        }
      }
    };

    /**
     * The main detection loop that continuously estimates poses and updates the canvas.
     */
    const detectLoop = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || !detectorRef.current) {
        rafRef.current = requestAnimationFrame(detectLoop);
        return;
      }

      const ctx = canvas.getContext('2d');
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      try {
        const start = performance.now();
        const poses = await detectorRef.current.estimatePoses(video, { flipHorizontal: false });
        const elapsed = performance.now() - start;

        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(video, 0, 0, width, height);

        const pose = poses?.[0];
        if (pose) {
          const selected = ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle'].map(
            (name) => getJoint(pose, name),
          );

          const leftHip = getJoint(pose, 'left_hip');
          const leftKnee = getJoint(pose, 'left_knee');
          const leftAnkle = getJoint(pose, 'left_ankle');
          const rightHip = getJoint(pose, 'right_hip');
          const rightKnee = getJoint(pose, 'right_knee');
          const rightAnkle = getJoint(pose, 'right_ankle');

          // Prefer left side, fallback right side, so users can rotate naturally.
          const candidate =
            leftHip && leftKnee && leftAnkle
              ? { a: leftHip, b: leftKnee, c: leftAnkle }
              : rightHip && rightKnee && rightAnkle
                ? { a: rightHip, b: rightKnee, c: rightAnkle }
                : null;

          if (candidate) {
            const rawAngle = calculateAngle(candidate.a, candidate.b, candidate.c);
            const angle = Math.round(rawAngle);
            const safeAngle = Number.isFinite(angle) ? angle : 180;

            setAngleValue(safeAngle);
            setFeedback(safeAngle > 150 ? 'Deep squat' : safeAngle > 100 ? 'Good depth' : 'Stand tall');

            drawLine(ctx, candidate.a, candidate.b, '#8b5cf6', 5);
            drawLine(ctx, candidate.b, candidate.c, '#8b5cf6', 5);
            drawJoint(ctx, candidate.a, '#34d399', 8);
            drawJoint(ctx, candidate.b, '#fbbf24', 8);
            drawJoint(ctx, candidate.c, '#f87171', 8);
            drawAngleArc(ctx, candidate.a, candidate.b, candidate.c, safeAngle);

            ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
            ctx.fillRect(16, 16, 220, 68);
            ctx.fillStyle = '#f8fafc';
            ctx.font = '600 20px sans-serif';
            ctx.fillText(`Angle: ${safeAngle}°`, 30, 46);
            ctx.font = '500 16px sans-serif';
            ctx.fillText(
              `Status: ${safeAngle > 150 ? 'Deep squat' : safeAngle > 100 ? 'Good depth' : 'Stand tall'}`,
              30,
              64,
            );
          } else {
            setFeedback('Position your body in frame');
          }

          const visiblePoints = selected.filter(Boolean);
          if (visiblePoints.length > 0) {
            const skeletonConnections = [
              ['left_hip', 'left_knee'],
              ['left_knee', 'left_ankle'],
              ['right_hip', 'right_knee'],
              ['right_knee', 'right_ankle'],
            ];

            skeletonConnections.forEach(([startName, endName]) => {
              const startPoint = getJoint(pose, startName);
              const endPoint = getJoint(pose, endName);
              if (startPoint && endPoint) {
                drawLine(ctx, startPoint, endPoint, '#60a5fa', 3);
              }
            });

            visiblePoints.forEach((point) => drawJoint(ctx, point, '#a78bfa', 6));
          }
        }

        const processedFps = Math.max(0, Math.round(1000 / Math.max(elapsed, 1)));
        setFps(processedFps);
      } catch (error) {
        console.error('Pose detection error:', error);
      }

      rafRef.current = requestAnimationFrame(detectLoop);
    };

    init();

    return () => {
      cancelled = true;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }

      const currentVideo = videoRef.current;
      if (currentVideo?.srcObject) {
        currentVideo.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
      <div className="mx-auto max-w-6xl">
        <header className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-300">Pose AI</p>
            <h1 className="mt-2 text-3xl font-bold text-white">Squat depth tracker</h1>
          </div>
          <div className="flex items-center gap-3 rounded-full border border-slate-700 bg-slate-900/80 px-4 py-2 text-sm text-slate-300">
            <span className={`h-2.5 w-2.5 rounded-full ${isReady ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            {status}
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_320px]">
          <div className="relative overflow-hidden rounded-3xl border border-slate-700 bg-slate-900 shadow-2xl shadow-violet-950/30">
            <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-950">
              <video ref={videoRef} className="h-full w-full object-cover" playsInline muted />
              <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-cover" />
            </div>
          </div>

          <aside className="space-y-4 rounded-3xl border border-slate-700 bg-slate-900 p-5 shadow-xl shadow-slate-950/40">
            <div className="rounded-2xl border border-violet-500/30 bg-violet-500/10 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-violet-200">Current measurement</p>
              <div className="mt-3 flex items-end gap-2">
                <span className="text-5xl font-black text-white">{angleValue}</span>
                <span className="pb-2 text-xl text-violet-200">°</span>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-800/80 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Form feedback</p>
              <p className="mt-3 text-2xl font-semibold text-white">{feedback}</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-slate-700 bg-slate-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">FPS</p>
                <p className="mt-2 text-2xl font-bold text-white">{fps}</p>
              </div>
              <div className="rounded-2xl border border-slate-700 bg-slate-800 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Target</p>
                <p className="mt-2 text-2xl font-bold text-white">90-110°</p>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-800 p-4 text-sm leading-6 text-slate-300">
              <p className="font-semibold text-white">Joint target</p>
              <p className="mt-2">Hip - Knee - Ankle</p>
              <p>Lower into the squat and watch the knee angle update live.</p>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}

export default App;
