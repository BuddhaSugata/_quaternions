import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import Papa from "papaparse";

/**
 * Unimetry 3D D-Rotation Simulator (React + Three.js)
 * ---------------------------------------------------
 * FIX: Robust CSV parsing with Papa.parse.
 * - Guard against empty/invalid results.data.
 * - Support common RA/Dec aliases (ra, ra_icrs, ra_deg / dec, dec_icrs, dec_deg).
 * - Disable worker for File/Blob inputs (avoids cloning issues in some browsers).
 * - Clear error messages.
 * - Add built-in parser test cases (run from UI) without changing any existing behavior.
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

// Map common column aliases to canonical keys
const RA_ALIASES = ["ra", "ra_icrs", "ra_deg", "rightascension", "ra_j2000", "_ra" ];
const DEC_ALIASES = ["dec", "dec_icrs", "dec_deg", "declination", "dec_j2000", "_dec" ];

function pickKeyCaseInsensitive(row) {
  if (!row || typeof row !== "object") return { raKey: null, decKey: null };
  const lowerToOrig = Object.fromEntries(Object.keys(row).map((k) => [k.toLowerCase().trim(), k]));
  let raKey = null, decKey = null;
  for (const a of RA_ALIASES) { if (lowerToOrig[a]) { raKey = lowerToOrig[a]; break; } }
  for (const a of DEC_ALIASES) { if (lowerToOrig[a]) { decKey = lowerToOrig[a]; break; } }
  return { raKey, decKey };
}

function parseCSVtoVecs(input, maxRows = 50000) {
  /**
   * Robust CSV parser:
   * - Accepts File/Blob/String.
   * - header:true, skipEmptyLines:true.
   * - worker disabled for File/Blob to avoid structured-clone quirks.
   */
  const isBlob = (typeof Blob !== "undefined" && input instanceof Blob);
  return new Promise((resolve, reject) => {
    Papa.parse(input, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: "greedy",
      worker: !isBlob, // avoid worker when input is a File/Blob
      complete: (results) => {
        try {
          const rows = Array.isArray(results?.data) ? results.data : [];
          if (!rows.length) return reject(new Error("CSV has no data rows (after skipping empties)."));

          // find a non-empty sample row
          const sample = rows.find((r) => r && Object.keys(r).length > 0);
          if (!sample) return reject(new Error("CSV rows are empty or malformed."));

          const { raKey, decKey } = pickKeyCaseInsensitive(sample);
          if (!raKey || !decKey) {
            return reject(new Error("CSV must include RA/Dec columns (deg). Accepted aliases: ra|ra_icrs|ra_deg and dec|dec_icrs|dec_deg."));
          }

          const out = [];
          for (let i = 0; i < rows.length && out.length / 3 < maxRows; i++) {
            const row = rows[i];
            if (!row) continue;
            const ra = Number(row[raKey]);
            const dec = Number(row[decKey]);
            if (!isFinite(ra) || !isFinite(dec)) continue; // skip bad lines
            const v = radecToVec(ra, dec);
            out.push(v[0], v[1], v[2]);
          }
          if (out.length === 0) return reject(new Error("No valid RA/Dec values found in CSV."));
          resolve(new Float32Array(out));
        } catch (e) {
          reject(e);
        }
      },
      error: (err) => reject(err),
    });
  });
}

function parseCSVString(text, maxRows = 50000) {
  // Helper for tests
  const blob = new Blob([text], { type: "text/csv" });
  return parseCSVtoVecs(blob, maxRows);
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
function StarPoints({ positions, size = 0.01 }) {
  const ref = useRef();
  const geomRef = useRef();

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

// ---------- Built-in parser tests ----------
function useParserTests() {
  const [lastRun, setLastRun] = useState(null);

  async function runTests() {
    const results = [];
    function pass(name) { results.push({ name, ok: true }); }
    function fail(name, err) { results.push({ name, ok: false, err: err?.message || String(err) }); }

    // Test 1: minimal valid headers ra,dec
    try {
      const t1 = await parseCSVString("ra,dec\n0,0\n180,0\n", 10);
      if (t1.length === 6) pass("T1: ra,dec minimal"); else fail("T1: ra,dec minimal", new Error("unexpected length"));
    } catch (e) { fail("T1: ra,dec minimal", e); }

    // Test 2: aliases ra_icrs,dec_icrs
    try {
      const t2 = await parseCSVString("ra_icrs,dec_icrs\n10,5\n20,-5\n", 10);
      if (t2.length === 6) pass("T2: aliases ra_icrs/dec_icrs"); else fail("T2: aliases", new Error("unexpected length"));
    } catch (e) { fail("T2: aliases ra_icrs/dec_icrs", e); }

    // Test 3: empty file -> error
    try {
      await parseCSVString("\n\n", 10);
      fail("T3: empty CSV should error", new Error("no error thrown"));
    } catch (_) { pass("T3: empty CSV error raised"); }

    setLastRun(results);
  }

  return { lastRun, runTests };
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
  const [message, setMessage] = useState("");

  const { lastRun, runTests } = useParserTests();

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
      setMessage("");
    } catch (err) {
      setMessage("CSV parse error: " + (err?.message || String(err)));
      console.error(err);
      alert("CSV parse error: " + (err?.message || String(err)));
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
          <button className="px-3 py-1 rounded bg-purple-600 hover:bg-purple-500" onClick={runTests}>Run parser tests</button>
        </div>
        {message && <p className="text-xs text-red-300">{message}</p>}
        {lastRun && (
          <div className="text-xs bg-white/10 rounded p-2 space-y-1">
            <div className="font-semibold">Parser tests</div>
            {lastRun.map((t, i) => (
              <div key={i} className={t.ok ? "text-emerald-300" : "text-rose-300"}>
                {t.ok ? "✓" : "✗"} {t.name}{!t.ok && t.err ? ` — ${t.err}` : ""}
              </div>
            ))}
          </div>
        )}
        <p className="text-xs opacity-80">
          CSV needs <code>ra, dec</code> in degrees (aliases supported). This viewer applies the real–imag D-rotation
          along axis <code>u</code> and shows transformed directions on the unit sphere.
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
