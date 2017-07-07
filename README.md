# cross-validation

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![npm download][download-image]][download-url]

Utility library to do cross validation with supervised classifiers.

Cross-validation methods: 
- [k-fold](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation)
- [leave-p-out](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-p-out_cross-validation)

[API documentation](https://mljs.github.io/cross-validation/).

A list of the mljs supervised classifiers is available [here](https://github.com/mljs/ml#tools) in the supervised learning section, but you could also use your own. Cross validations methods return a ConfusionMatrix ([https://github.com/mljs/confusion-matrix](https://github.com/mljs/confusion-matrix) that can be used to calculate metrics on your classification result.

## Installation
```bash
npm i -s ml-cross-validation
```

## Example
```js
const crossValidation = require('ml-cross-validation');
const KNN = require('ml-knn');
const dataset = [[0, 0, 0], [0, 1, 1], [1, 1, 0], [2, 2, 2], [1, 2, 2], [2, 1, 2]];
const labels = [0, 0, 0, 1, 1, 1];
const confusionMatrix = crossValidation.leaveOneOut(KNN, dataSet, labels);
const accuracy = confusionMatrix.getAccuracy();
```

## Use your own classification library with ml-cross-validation
To be used with ml-cross-validation, your classification library must implement
- A constructor. The constructor can be passed options as a single argument.
- A `train` method. The `train` method is passed the data as a first argument and the labels as a second.
- A `predict` method. The `predict` method is passed test data and should return a predicted label.

```js
class MyClassifier {
  constructor(options) {
    this.options = options;
  }
  train(data, labels) {
    // Create your model
  }
  predict(testData) {
    // Apply your model and return predicted label
    return prediction;
  }
}
```

[npm-image]: https://img.shields.io/npm/v/ml-cross-validation.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-cross-validation
[travis-image]: https://img.shields.io/travis/mljs/cross-validation/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/cross-validation
[download-image]: https://img.shields.io/npm/dm/ml-cross-validation.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-cross-validation
