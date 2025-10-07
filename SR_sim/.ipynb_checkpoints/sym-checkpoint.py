import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import Papa from "papaparse";

/**
 * Unimetry 3D D-Rotation Simulator
 * --------------------------------
 * - Interactive 3D starfield with quaternionic D-rotation (real-imaginary) along axis u.
 * - Load your own CSV (columns: ra, dec in degrees) or generate synthetic all-sky.
 * - Controls: axis (RA/Dec), beta in [0, 0.99), sample size, play/pause acceleration.
 * - Export transformed sky (ra', dec').
 *
 * Math (Im H ≅ R^3): for unit direction n and axis u, with tan(alpha)=beta:
 *   n' = n_parallel + cos(alpha) * n_perp, then normalize to unit.
 *   where n_parallel = (n·u) u, n_perp = n - n_parallel.
 * cos(alpha) = 1/gamma, sin(alpha) = beta/gamma.
 */

// ---------- Math helpers ----------
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function unit(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1e-15;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function radecToVec(raDeg, decDeg) {
  const ra = raDeg * DEG2RAD;
  const dec = decDeg * DEG2RAD;
  const x = Math.cos(dec) * Math.cos(ra);
  const y = Math.cos(dec) * Math.sin(ra);
  const z = Math.sin(dec);
  return [x, y, z];
}

function vecToRadec(v) {
  const [x, y, z] = v;
  const ra = (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI);
  const dec = Math.asin(Math.max(-1, Math.min(1, z)));
  return [ra * RAD2DEG, dec * RAD2DEG];
}

function alphaFromBeta(beta) {
  return Math.atan(Math.max(-0.999999, Math.min(0.999999, beta)));
}

function dRotateVec(n, u, beta) {
  // n, u are unit arrays [x,y,z]
  const alpha = alphaFromBeta(beta);
  const c = Math.cos(alpha);
  const dot = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
  const nPar = [dot * u[0], dot * u[1], dot * u[2]];
  const nPerp = [n[0] - nPar[0], n[1] - nPar[1], n[2] - nPar[2]];
  const out = [nPar[0] + c * nPerp[0], nPar[1] + c * nPerp[1], nPar[2] + c * nPerp[2]];
  return unit(out);
}

// ---------- Data generation & parsing ----------
function generateUniformSphere(N, seed = 42) {
  // Fast uniform directions on S^2
  const rng = mulberry32(seed);
  const out = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const u = 2 * rng() - 1; // cos(dec)
    const phi = 2 * Math.PI * rng();
    const s = Math.sqrt(1 - u * u);
    const x = s * Math.cos(phi);
    const y = s * Math.sin(phi);
    const z = u;
    out[3 * i + 0] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
  }
  return out;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function parseCSVtoVecs(file, maxRows = 50000) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      worker: true,
      step: undefined,
      complete: (results) => {
        try {
          const data = results.data;
          const keys = data.length ? Object.keys(data[0]).map((k) => k.toLowerCase()) : [];
          const raKey = (data.length && Object.keys(data[0]).find((k) => k.toLowerCase() === "ra")) || null;
          const decKey = (data.length && Object.keys(data[0]).find((k) => k.toLowerCase() === "dec")) || null;
          if (!raKey || !decKey) return reject(new Error("CSV must have 'ra' and 'dec' columns (degrees)."));
          const N = Math.min(maxRows, data.length);
          const out = new Float32Array(N * 3);
          for (let i = 0; i < N; i++) {
            const row = data[i];
            const ra = Number(row[raKey]);
            const dec = Number(row[decKey]);
            if (!isFinite(ra) || !isFinite(dec)) continue;
            const v = radecToVec(ra, dec);
            out[3 * i + 0] = v[0];
            out[3 * i + 1] = v[1];
            out[3 * i + 2] = v[2];
          }
          resolve(out);
        } catch (e) {
          reject(e);
        }
      },
      error: (err) => reject(err),
    });
  });
}

function exportTransformedCSV(positions) {
  const N = positions.length / 3;
  const rows = new Array(N + 1);
  rows[0] = "ra_prime,dec_prime";
  for (let i = 0; i < N; i++) {
    const v = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
    const [ra, dec] = vecToRadec(unit(v));
    rows[i + 1] = `${ra.toFixed(6)},${dec.toFixed(6)}`;
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sky_transformed.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------- Three.js components ----------
function StarPoints({ positions, size = 0.01, color = "white" }) {
  const ref = useRef();
  const geomRef = useRef();

  // Set or update positions buffer
  useEffect(() => {
    if (!geomRef.current) return;
    const geom = geomRef.current;
    const attr = new THREE.Float32BufferAttribute(positions, 3);
    geom.setAttribute("position", attr);
    geom.computeBoundingSphere();
  }, [positions]);

  return (
    <points ref={ref} frustumCulled={true}>
      <bufferGeometry ref={geomRef} />
      <pointsMaterial size={size} sizeAttenuation depthWrite={false} />
    </points>
  );
}

function AxisArrow({ u, length = 1.3, color = "#ff5050" }) {
  const points = useMemo(() => {
    const dir = unit(u);
    return [new THREE.Vector3(0, 0, 0), new THREE.Vector3(dir[0] * length, dir[1] * length, dir[2] * length)];
  }, [u, length]);
  const geom = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  return (
    <line>
      <primitive object={geom} />
      <lineBasicMaterial linewidth={2} color={color} />
    </line>
  );
}

function SphereWire({ radius = 1, segments = 32, color = "#888" }) {
  const geom = useMemo(() => new THREE.SphereGeometry(radius, segments, segments), [radius, segments]);
  return (
    <mesh>
      <primitive object={geom} />
      <meshBasicMaterial wireframe color={color} transparent opacity={0.3} />
    </mesh>
  );
}

// ---------- Main App ----------
export default function Unimetry3DSim() {
  const [N, setN] = useState(30000);
  const [positions, setPositions] = useState(() => generateUniformSphere(30000));
  const [beta, setBeta] = useState(0.6);
  const [uRA, setURA] = useState(0);
  const [uDec, setUDec] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [pointSize, setPointSize] = useState(0.01);

  const u = useMemo(() => unit(radecToVec(uRA, uDec)), [uRA, uDec]);

  const transformed = useMemo(() => {
    const alpha = alphaFromBeta(beta);
    const c = Math.cos(alpha);
    const out = new Float32Array(positions.length);
    const ux = u[0], uy = u[1], uz = u[2];
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i], y = positions[i + 1], z = positions[i + 2];
      const dot = x * ux + y * uy + z * uz;
      const px = dot * ux, py = dot * uy, pz = dot * uz; // parallel
      const vx = x - px, vy = y - py, vz = z - pz; // perp
      const nx = px + c * vx, ny = py + c * vy, nz = pz + c * vz;
      const inv = 1 / Math.max(1e-15, Math.hypot(nx, ny, nz));
      out[i] = nx * inv; out[i + 1] = ny * inv; out[i + 2] = nz * inv;
    }
    return out;
  }, [positions, u, beta]);

  // Animation loop to ramp beta
  const betaRef = useRef(beta);
  useEffect(() => { betaRef.current = beta; }, [beta]);

  function AnimatedBeta() {
    useFrame((state, dt) => {
      if (!playing) return;
      let b = betaRef.current + dt * 0.05; // accelerate slowly
      if (b > 0.98) b = 0.0;
      setBeta(b);
    });
    return null;
  }

  // Handlers
  function regenerate() {
    const arr = generateUniformSphere(N);
    setPositions(arr);
  }

  async function onCSV(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const vecs = await parseCSVtoVecs(file, 120000);
      setPositions(vecs);
      setN(vecs.length / 3);
    } catch (err) {
      alert("CSV parse error: " + err.message);
    }
  }

  function onExport() {
    exportTransformedCSV(transformed);
  }

  return (
    <div className="w-full h-full relative">
      {/* UI overlay */}
      <div className="absolute top-2 left-2 z-10 bg-black/60 text-white rounded-xl p-3 space-y-2 max-w-md">
        <div className="font-semibold">Unimetry 3D D-Rotation</div>
        <div className="grid grid-cols-2 gap-2 items-center">
          <label>β (speed): {beta.toFixed(2)}</label>
          <input type="range" min={0} max={0.99} step={0.001} value={beta} onChange={(e) => setBeta(parseFloat(e.target.value))} />

          <label>Axis RA (°): {uRA.toFixed(1)}</label>
          <input type="range" min={0} max={360} step={0.1} value={uRA} onChange={(e) => setURA(parseFloat(e.target.value))} />

          <label>Axis Dec (°): {uDec.toFixed(1)}</label>
          <input type="range" min={-90} max={90} step={0.1} value={uDec} onChange={(e) => setUDec(parseFloat(e.target.value))} />

          <label>Points: {N.toLocaleString()}</label>
          <input type="range" min={1000} max={120000} step={1000} value={N} onChange={(e) => setN(parseInt(e.target.value))} onMouseUp={regenerate} onTouchEnd={regenerate} />

          <label>Point size: {pointSize.toFixed(3)}</label>
          <input type="range" min={0.003} max={0.03} step={0.001} value={pointSize} onChange={(e) => setPointSize(parseFloat(e.target.value))} />
        </div>
        <div className="flex gap-2 flex-wrap">
          <button className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button className="px-3 py-1 rounded bg-sky-600 hover:bg-sky-500" onClick={regenerate}>Regenerate</button>
          <label className="px-3 py-1 rounded bg-zinc-700 cursor-pointer hover:bg-zinc-600">
            Load CSV
            <input type="file" accept=".csv" className="hidden" onChange={onCSV} />
          </label>
          <button className="px-3 py-1 rounded bg-amber-600 hover:bg-amber-500" onClick={onExport}>Export transformed CSV</button>
        </div>
        <p className="text-xs opacity-80">
          CSV columns required: <code>ra, dec</code> (degrees). This viewer applies D-rotation in the real–imag plane
          along axis <code>u</code> and shows the transformed directions on the unit sphere.
        </p>
      </div>

      {/* 3D Canvas */}
      <Canvas camera={{ position: [0, 0, 3], fov: 60 }}>
        <color attach="background" args={["#050505"]} />
        <ambientLight intensity={0.7} />
        <pointLight position={[3, 3, 3]} intensity={0.5} />

        {/* Sphere grid + axes */}
        <SphereWire radius={1.0} segments={32} />
        <AxisArrow u={u} />

        {/* Starfield */}
        <StarPoints positions={transformed} size={pointSize} />

        <OrbitControls enablePan={false} enableDamping dampingFactor={0.08} rotateSpeed={0.6} />
        <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
          <GizmoViewport axisColors={["#f55", "#5f5", "#55f"]} labelColor="white" />
        </GizmoHelper>
        <AnimatedBeta />
      </Canvas>
    </div>
  );
}
