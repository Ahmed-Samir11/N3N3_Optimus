import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '2mb',
    },
    responseLimit: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { Q, k, candidates } = req.body;
  if (!Q || !k || !candidates) return res.status(400).json({ error: 'Missing Q, k, or candidates' });

  // Write QUBO to a temp file
  const quboPath = path.join('/tmp', `qubo_${Date.now()}.json`);
  fs.writeFileSync(quboPath, JSON.stringify(Q));

  // Call quantum_optimize.py
  const pyPath = path.join(process.cwd(), 'quantum_optimize.py');
  const args = ['--qubo_file', quboPath, '--k', String(k)];
  const start = Date.now();
  const py = spawn('python3', [pyPath, ...args]);
  let out = '', err = '';
  let responded = false;
  const timeout = setTimeout(() => {
    if (!responded) {
      responded = true;
      py.kill('SIGKILL');
      fs.unlinkSync(quboPath);
      console.error('Quantum optimization timed out.\nSTDOUT:', out, '\nSTDERR:', err);
      res.status(500).json({ error: 'Quantum optimization timed out', stdout: out, stderr: err });
    }
  }, 60000); // 60s timeout
  py.stdout.on('data', d => { out += d.toString(); });
  py.stderr.on('data', d => { err += d.toString(); });
  py.on('close', code => {
    if (responded) return;
    responded = true;
    clearTimeout(timeout);
    fs.unlinkSync(quboPath);
    if (code !== 0) {
      console.error('Quantum optimization failed.\nSTDOUT:', out, '\nSTDERR:', err);
      res.status(500).json({ error: 'Quantum optimization failed', details: err, stdout: out, code });
      return;
    }
    try {
      // Defensive: find last valid JSON in output
      const lines = out.trim().split('\n');
      let result = null;
      for (let i = lines.length - 1; i >= 0; i--) {
        try {
          result = JSON.parse(lines[i]);
          break;
        } catch (e) {}
      }
      if (!result) throw new Error('No valid JSON output from quantum_optimize.py');
      result.selected_grids = (result.selected_indices || []).map(i => candidates[i]);
      result.time_ms = Date.now() - start;
      res.status(200).json({ grids: result.selected_grids || [], time_ms: result.time_ms, raw: result });
    } catch (e) {
      console.error('Failed to parse quantum output.\nSTDOUT:', out, '\nSTDERR:', err, '\nException:', e);
      res.status(500).json({ error: 'Failed to parse quantum output', details: out + err, exception: e.toString() });
    }
  });
}
