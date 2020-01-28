import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { getClasses } from 'ml-dataset-iris';

import { sampleAClass } from '../sampleAClass.js';

expect.extend({ toBeDeepCloseTo });

const metadata = getClasses();

describe('get sample of class', () => {
  it('check length', () => {
    expect(metadata).toHaveLength(150);
  });
  let c = sampleAClass(metadata, 0.1).trainIndex;
  it('check length', () => {
    expect(c).toHaveLength(15);
  });
});
