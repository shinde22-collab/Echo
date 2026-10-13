import { readFileSync } from 'node:fs';
import { predict } from '../docs/inference.mjs';

const data = JSON.parse(readFileSync(new URL('../docs/assets/experiment.json', import.meta.url), 'utf8'));
let count = 0;
for (const run of data.runs) {
  for (let i = 0; i < data.test.x.length; i++) {
    const result = predict(run.model, data.test.x[i], data.scaler);
    if (result !== run.predictions[i]) throw new Error(`Prediction mismatch: ${run.strategy}/${run.seed}/${run.budget}, sample ${i}`);
    count++;
  }
}
console.log(`${count} browser-model predictions match Scikit-learn across all 90 checkpoints.`);
