export default {
  input: 'src/index.js',
  output: {
    format: 'cjs',
    file: 'lib/index.js',
  },
  external: [
    'jest-matcher-deep-close-to',
    'ml-combinations',
    'ml-dataset-iris',
    'ml-confusion-matrix',
  ],
};
