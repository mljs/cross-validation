/**
 * ml-cross-validation - Cross validation utility for mljs classifiers
 * @version v1.2.0
 * @link https://github.com/mljs/cross-validation#readme
 * @license MIT
 */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = global || self, factory(global.mlCrossValidation = {}));
}(this, function (exports) { 'use strict';

  /**
   *  Constructs a confusion matrix
   * @class ConfusionMatrix
   * @example
   * const CM = new ConfusionMatrix([[13, 2], [10, 5]], ['cat', 'dog'])
   * @param {Array<Array<number>>} matrix - The confusion matrix, a 2D Array. Rows represent the actual label and columns
   *     the predicted label.
   * @param {Array<any>} labels - Labels of the confusion matrix, a 1D Array
   */
  class ConfusionMatrix {
    constructor(matrix, labels) {
      if (matrix.length !== matrix[0].length) {
        throw new Error('Confusion matrix must be square');
      }

      if (labels.length !== matrix.length) {
        throw new Error('Confusion matrix and labels should have the same length');
      }

      this.labels = labels;
      this.matrix = matrix;
    }
    /**
     * Construct confusion matrix from the predicted and actual labels (classes). Be sure to provide the arguments in
     * the correct order!
     * @param {Array<any>} actual  - The predicted labels of the classification
     * @param {Array<any>} predicted     - The actual labels of the classification. Has to be of same length as
     *     predicted.
     * @param {object} [options] - Additional options
     * @param {Array<any>} [options.labels] - The list of labels that should be used. If not provided the distinct set
     *     of labels present in predicted and actual is used. Labels are compared using the strict equality operator
     *     '==='
     * @return {ConfusionMatrix} - Confusion matrix
     */


    static fromLabels(actual, predicted) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

      if (predicted.length !== actual.length) {
        throw new Error('predicted and actual must have the same length');
      }

      let distinctLabels;

      if (options.labels) {
        distinctLabels = new Set(options.labels);
      } else {
        distinctLabels = new Set([...actual, ...predicted]);
      }

      distinctLabels = Array.from(distinctLabels);

      if (options.sort) {
        distinctLabels.sort(options.sort);
      } // Create confusion matrix and fill with 0's


      const matrix = Array.from({
        length: distinctLabels.length
      });

      for (let i = 0; i < matrix.length; i++) {
        matrix[i] = new Array(matrix.length);
        matrix[i].fill(0);
      }

      for (let i = 0; i < predicted.length; i++) {
        const actualIdx = distinctLabels.indexOf(actual[i]);
        const predictedIdx = distinctLabels.indexOf(predicted[i]);

        if (actualIdx >= 0 && predictedIdx >= 0) {
          matrix[actualIdx][predictedIdx]++;
        }
      }

      return new ConfusionMatrix(matrix, distinctLabels);
    }
    /**
     * Get the confusion matrix
     * @return {Array<Array<number> >}
     */


    getMatrix() {
      return this.matrix;
    }

    getLabels() {
      return this.labels;
    }
    /**
     * Get the total number of samples
     * @return {number}
     */


    getTotalCount() {
      let predicted = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          predicted += this.matrix[i][j];
        }
      }

      return predicted;
    }
    /**
     * Get the total number of true predictions
     * @return {number}
     */


    getTrueCount() {
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        count += this.matrix[i][i];
      }

      return count;
    }
    /**
     * Get the total number of false predictions.
     * @return {number}
     */


    getFalseCount() {
      return this.getTotalCount() - this.getTrueCount();
    }
    /**
     * Get the number of true positive predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTruePositiveCount(label) {
      const index = this.getIndex(label);
      return this.matrix[index][index];
    }
    /**
     * Get the number of true negative predictions
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTrueNegativeCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          if (i !== index && j !== index) {
            count += this.matrix[i][j];
          }
        }
      }

      return count;
    }
    /**
     * Get the number of false positive predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalsePositiveCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        if (i !== index) {
          count += this.matrix[i][index];
        }
      }

      return count;
    }
    /**
     * Get the number of false negative predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseNegativeCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        if (i !== index) {
          count += this.matrix[index][i];
        }
      }

      return count;
    }
    /**
     * Get the number of real positive samples.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getPositiveCount(label) {
      return this.getTruePositiveCount(label) + this.getFalseNegativeCount(label);
    }
    /**
     * Get the number of real negative samples.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getNegativeCount(label) {
      return this.getTrueNegativeCount(label) + this.getFalsePositiveCount(label);
    }
    /**
     * Get the index in the confusion matrix that corresponds to the given label
     * @param {any} label - The label to search for
     * @throws if the label is not found
     * @return {number}
     */


    getIndex(label) {
      const index = this.labels.indexOf(label);
      if (index === -1) throw new Error('The label does not exist');
      return index;
    }
    /**
     * Get the true positive rate a.k.a. sensitivity. Computes the ratio between the number of true positive predictions and the total number of positive samples.
     * {@link https://en.wikipedia.org/wiki/Sensitivity_and_specificity}
     * @param {any} label - The label that should be considered "positive"
     * @return {number} - The true positive rate [0-1]
     */


    getTruePositiveRate(label) {
      return this.getTruePositiveCount(label) / this.getPositiveCount(label);
    }
    /**
     * Get the true negative rate a.k.a. specificity. Computes the ration between the number of true negative predictions and the total number of negative samples.
     * {@link https://en.wikipedia.org/wiki/Sensitivity_and_specificity}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTrueNegativeRate(label) {
      return this.getTrueNegativeCount(label) / this.getNegativeCount(label);
    }
    /**
     * Get the positive predictive value a.k.a. precision. Computes TP / (TP + FP)
     * {@link https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getPositivePredictiveValue(label) {
      const TP = this.getTruePositiveCount(label);
      return TP / (TP + this.getFalsePositiveCount(label));
    }
    /**
     * Negative predictive value
     * {@link https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getNegativePredictiveValue(label) {
      const TN = this.getTrueNegativeCount(label);
      return TN / (TN + this.getFalseNegativeCount(label));
    }
    /**
     * False negative rate a.k.a. miss rate.
     * {@link https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseNegativeRate(label) {
      return 1 - this.getTruePositiveRate(label);
    }
    /**
     * False positive rate a.k.a. fall-out rate.
     * {@link https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalsePositiveRate(label) {
      return 1 - this.getTrueNegativeRate(label);
    }
    /**
     * False discovery rate (FDR)
     * {@link https://en.wikipedia.org/wiki/False_discovery_rate}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseDiscoveryRate(label) {
      const FP = this.getFalsePositiveCount(label);
      return FP / (FP + this.getTruePositiveCount(label));
    }
    /**
     * False omission rate (FOR)
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseOmissionRate(label) {
      const FN = this.getFalseNegativeCount(label);
      return FN / (FN + this.getTruePositiveCount(label));
    }
    /**
     * F1 score
     * {@link https://en.wikipedia.org/wiki/F1_score}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getF1Score(label) {
      const TP = this.getTruePositiveCount(label);
      return 2 * TP / (2 * TP + this.getFalsePositiveCount(label) + this.getFalseNegativeCount(label));
    }
    /**
     * Matthews correlation coefficient (MCC)
     * {@link https://en.wikipedia.org/wiki/Matthews_correlation_coefficient}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getMatthewsCorrelationCoefficient(label) {
      const TP = this.getTruePositiveCount(label);
      const TN = this.getTrueNegativeCount(label);
      const FP = this.getFalsePositiveCount(label);
      const FN = this.getFalseNegativeCount(label);
      return (TP * TN - FP * FN) / Math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
    }
    /**
     * Informedness
     * {@link https://en.wikipedia.org/wiki/Youden%27s_J_statistic}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getInformedness(label) {
      return this.getTruePositiveRate(label) + this.getTrueNegativeRate(label) - 1;
    }
    /**
     * Markedness
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getMarkedness(label) {
      return this.getPositivePredictiveValue(label) + this.getNegativePredictiveValue(label) - 1;
    }
    /**
     * Get the confusion table.
     * @param {any} label - The label that should be considered "positive"
     * @return {Array<Array<number> >} - The 2x2 confusion table. [[TP, FN], [FP, TN]]
     */


    getConfusionTable(label) {
      return [[this.getTruePositiveCount(label), this.getFalseNegativeCount(label)], [this.getFalsePositiveCount(label), this.getTrueNegativeCount(label)]];
    }
    /**
     * Get total accuracy.
     * @return {number} - The ratio between the number of true predictions and total number of classifications ([0-1])
     */


    getAccuracy() {
      let correct = 0;
      let incorrect = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          if (i === j) correct += this.matrix[i][j];else incorrect += this.matrix[i][j];
        }
      }

      return correct / (correct + incorrect);
    }
    /**
     * Returns the element in the confusion matrix that corresponds to the given actual and predicted labels.
     * @param {any} actual - The true label
     * @param {any} predicted - The predicted label
     * @return {number} - The element in the confusion matrix
     */


    getCount(actual, predicted) {
      const actualIndex = this.getIndex(actual);
      const predictedIndex = this.getIndex(predicted);
      return this.matrix[actualIndex][predictedIndex];
    }
    /**
     * Compute the general prediction accuracy
     * @deprecated Use getAccuracy
     * @return {number} - The prediction accuracy ([0-1]
     */


    get accuracy() {
      return this.getAccuracy();
    }
    /**
     * Compute the number of predicted observations
     * @deprecated Use getTotalCount
     * @return {number}
     */


    get total() {
      return this.getTotalCount();
    }

  }

  const defaultOptions = {
    mode: 'index'
  };

  var src = function* src(M, N, options) {
    options = Object.assign({}, defaultOptions, options);
    var a = new Array(N);
    var c = new Array(M);
    var b = new Array(N);
    var p = new Array(N + 2);
    var x, y, z; // init a and b

    for (var i = 0; i < N; i++) {
      a[i] = i;
      if (i < N - M) b[i] = 0;else b[i] = 1;
    } // init c


    for (i = 0; i < M; i++) {
      c[i] = N - M + i;
    } // init p


    for (i = 0; i < p.length; i++) {
      if (i === 0) p[i] = N + 1;else if (i <= N - M) p[i] = 0;else if (i <= N) p[i] = i - N + M;else p[i] = -2;
    }

    function twiddle() {
      var i, j, k;
      j = 1;

      while (p[j] <= 0) {
        j++;
      }

      if (p[j - 1] === 0) {
        for (i = j - 1; i !== 1; i--) {
          p[i] = -1;
        }

        p[j] = 0;
        x = z = 0;
        p[1] = 1;
        y = j - 1;
      } else {
        if (j > 1) {
          p[j - 1] = 0;
        }

        do {
          j++;
        } while (p[j] > 0);

        k = j - 1;
        i = j;

        while (p[i] === 0) {
          p[i++] = -1;
        }

        if (p[i] === -1) {
          p[i] = p[k];
          z = p[k] - 1;
          x = i - 1;
          y = k - 1;
          p[k] = -1;
        } else {
          if (i === p[0]) {
            return 0;
          } else {
            p[j] = p[i];
            z = p[i] - 1;
            p[i] = 0;
            x = j - 1;
            y = i - 1;
          }
        }
      }

      return 1;
    }

    if (options.mode === 'index') {
      yield c.slice();

      while (twiddle()) {
        c[z] = a[x];
        yield c.slice();
      }
    } else if (options.mode === 'mask') {
      yield b.slice();

      while (twiddle()) {
        b[x] = 1;
        b[y] = 0;
        yield b.slice();
      }
    } else {
      throw new Error('Invalid mode');
    }
  };

  /**
   * get folds indexes
   * @param {Array} features
   * @param {Number} k - number of folds, a
   */
  function getFolds(features) {
    let k = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 5;
    let N = features.length;
    let allIdx = new Array(N);

    for (let i = 0; i < N; i++) {
      allIdx[i] = i;
    }

    let l = Math.floor(N / k); // create random k-folds

    let current = [];
    let folds = [];

    while (allIdx.length) {
      let randi = Math.floor(Math.random() * allIdx.length);
      current.push(allIdx[randi]);
      allIdx.splice(randi, 1);

      if (current.length === l) {
        folds.push(current);
        current = [];
      }
    } // we push the remaining to the last fold so that the total length is
    // preserved. Otherwise the Q2 will fail.


    if (current.length) current.forEach(e => folds[k - 1].push(e));
    folds = folds.slice(0, k);
    let foldsIndex = folds.map((x, idx) => ({
      testIndex: x,
      trainIndex: [].concat(...folds.filter((el, idx2) => idx2 !== idx))
    }));
    return foldsIndex;
  }

  /**
   * A function to sample a dataset maintaining classes equilibrated
   * @param {Array} classVector - an array containing class or group information
   * @param {Number} fraction - a fraction of the class to sample
   * @return {Object} - an object with indexes
   */
  function sampleAClass(classVector, fraction) {
    // sort the vector
    let classVectorSorted = JSON.parse(JSON.stringify(classVector));
    let result = Array.from(Array(classVectorSorted.length).keys()).sort((a, b) => classVectorSorted[a] < classVectorSorted[b] ? -1 : classVectorSorted[b] < classVectorSorted[a] | 0);
    classVectorSorted.sort((a, b) => a < b ? -1 : b < a | 0); // counts the class elements

    let counts = {};
    classVectorSorted.forEach(x => counts[x] = (counts[x] || 0) + 1); // pick a few per class

    let indexOfSelected = [];
    Object.keys(counts).forEach((e, i) => {
      let shift = [];
      Object.values(counts).reduce((a, c, item) => shift[item] = a + c, 0);
      let arr = [...Array(counts[e]).keys()];
      let r = [];

      for (let j = 0; j < Math.floor(counts[e] * fraction); j++) {
        let n = arr[Math.floor(Math.random() * arr.length)];
        r.push(n);
        let ind = arr.indexOf(n);
        arr.splice(ind, 1);
      }

      if (i === 0) {
        indexOfSelected = indexOfSelected.concat(r);
      } else {
        indexOfSelected = indexOfSelected.concat(r.map(x => x + shift[i - 1]));
      }
    }); // sort back the index

    let trainIndex = [];
    indexOfSelected.forEach(e => trainIndex.push(result[e]));
    let testIndex = [];
    let mask = [];
    classVector.forEach((el, idx) => {
      if (trainIndex.includes(idx)) {
        mask.push(true);
      } else {
        mask.push(false);
        testIndex.push(idx);
      }
    });
    return {
      trainIndex,
      testIndex,
      mask
    };
  }

  /**
   * Performs a leave-one-out cross-validation (LOO-CV) of the given samples. In LOO-CV, 1 observation is used as the
   * validation set while the rest is used as the training set. This is repeated once for each observation. LOO-CV is a
   * special case of LPO-CV. @see leavePout
   * @param {function} Classifier - The classifier's constructor to use for the cross validation. Expect ml-classifier
   *     api.
   * @param {Array} features - The features for all samples of the data-set
   * @param {Array} labels - The classification class of all samples of the data-set
   * @param {object} classifierOptions - The classifier options with which the classifier should be instantiated.
   * @return {ConfusionMatrix} - The cross-validation confusion matrix
   */

  function leaveOneOut(Classifier, features, labels, classifierOptions) {
    if (typeof labels === 'function') {
      let callback = labels;
      labels = features;
      features = Classifier;
      return leavePOut(features, labels, 1, callback);
    }

    return leavePOut(Classifier, features, labels, classifierOptions, 1);
  }
  /**
   * Performs a leave-p-out cross-validation (LPO-CV) of the given samples. In LPO-CV, p observations are used as the
   * validation set while the rest is used as the training set. This is repeated as many times as there are possible
   * ways to combine p observations from the set (unordered without replacement). Be aware that for relatively small
   * data-set size this can require a very large number of training and testing to do!
   * @param {function} Classifier - The classifier's constructor to use for the cross validation. Expect ml-classifier
   *     api.
   * @param {Array} features - The features for all samples of the data-set
   * @param {Array} labels - The classification class of all samples of the data-set
   * @param {object} classifierOptions - The classifier options with which the classifier should be instantiated.
   * @param {number} p - The size of the validation sub-samples' set
   * @return {ConfusionMatrix} - The cross-validation confusion matrix
   */

  function leavePOut(Classifier, features, labels, classifierOptions, p) {
    let callback;

    if (typeof classifierOptions === 'function') {
      callback = classifierOptions;
      p = labels;
      labels = features;
      features = Classifier;
    }

    check(features, labels);
    const distinct = getDistinct(labels);
    const confusionMatrix = initMatrix(distinct.length, distinct.length);
    let N = features.length;
    let gen = src(p, N);
    let allIdx = new Array(N);

    for (let i = 0; i < N; i++) {
      allIdx[i] = i;
    }

    for (const testIdx of gen) {
      let trainIdx = allIdx.slice();

      for (let i = testIdx.length - 1; i >= 0; i--) {
        trainIdx.splice(testIdx[i], 1);
      }

      if (callback) {
        validateWithCallback(features, labels, testIdx, trainIdx, confusionMatrix, distinct, callback);
      } else {
        validate(Classifier, features, labels, classifierOptions, testIdx, trainIdx, confusionMatrix, distinct);
      }
    }

    return new ConfusionMatrix(confusionMatrix, distinct);
  }
  /**
   * Performs k-fold cross-validation (KF-CV). KF-CV separates the data-set into k random equally sized partitions, and
   * uses each as a validation set, with all other partitions used in the training set. Observations left over from if k
   * does not divide the number of observations are left out of the cross-validation process.
   * @param {function} Classifier - The classifier's to use for the cross validation. Expect ml-classifier api.
   * @param {Array} features - The features for all samples of the data-set
   * @param {Array} labels - The classification class of all samples of the data-set
   * @param {object} classifierOptions - The classifier options with which the classifier should be instantiated.
   * @param {number} k - The number of partitions to create
   * @return {ConfusionMatrix} - The cross-validation confusion matrix
   */

  function kFold(Classifier, features, labels, classifierOptions, k) {
    let callback;

    if (typeof classifierOptions === 'function') {
      callback = classifierOptions;
      k = labels;
      labels = features;
      features = Classifier;
    }

    check(features, labels);
    const distinct = getDistinct(labels);
    const confusionMatrix = initMatrix(distinct.length, distinct.length);
    let folds = getFolds(features, k);

    for (let i = 0; i < folds.length; i++) {
      let testIdx = folds[i].testIndex;
      let trainIdx = folds[i].trainIndex;

      if (callback) {
        validateWithCallback(features, labels, testIdx, trainIdx, confusionMatrix, distinct, callback);
      } else {
        validate(Classifier, features, labels, classifierOptions, testIdx, trainIdx, confusionMatrix, distinct);
      }
    }

    return new ConfusionMatrix(confusionMatrix, distinct);
  }

  function check(features, labels) {
    if (features.length !== labels.length) {
      throw new Error('features and labels should have the same length');
    }
  }

  function initMatrix(rows, columns) {
    return new Array(rows).fill(0).map(() => new Array(columns).fill(0));
  }

  function getDistinct(arr) {
    let s = new Set();

    for (let i = 0; i < arr.length; i++) {
      s.add(arr[i]);
    }

    return Array.from(s);
  }

  function validate(Classifier, features, labels, classifierOptions, testIdx, trainIdx, confusionMatrix, distinct) {
    const {
      testFeatures,
      trainFeatures,
      testLabels,
      trainLabels
    } = getTrainTest(features, labels, testIdx, trainIdx);
    let classifier;

    if (Classifier.prototype.train) {
      classifier = new Classifier(classifierOptions);
      classifier.train(trainFeatures, trainLabels);
    } else {
      classifier = new Classifier(trainFeatures, trainLabels, classifierOptions);
    }

    let predictedLabels = classifier.predict(testFeatures);
    updateConfusionMatrix(confusionMatrix, testLabels, predictedLabels, distinct);
  }

  function validateWithCallback(features, labels, testIdx, trainIdx, confusionMatrix, distinct, callback) {
    const {
      testFeatures,
      trainFeatures,
      testLabels,
      trainLabels
    } = getTrainTest(features, labels, testIdx, trainIdx);
    const predictedLabels = callback(trainFeatures, trainLabels, testFeatures);
    updateConfusionMatrix(confusionMatrix, testLabels, predictedLabels, distinct);
  }

  function updateConfusionMatrix(confusionMatrix, testLabels, predictedLabels, distinct) {
    for (let i = 0; i < predictedLabels.length; i++) {
      const actualIdx = distinct.indexOf(testLabels[i]);
      const predictedIdx = distinct.indexOf(predictedLabels[i]);

      if (actualIdx < 0 || predictedIdx < 0) {
        // eslint-disable-next-line no-console
        console.warn("ignore unknown predicted label ".concat(predictedLabels[i]));
      }

      confusionMatrix[actualIdx][predictedIdx]++;
    }
  }

  function getTrainTest(features, labels, testIdx, trainIdx) {
    return {
      testFeatures: testIdx.map(function (index) {
        return features[index];
      }),
      trainFeatures: trainIdx.map(function (index) {
        return features[index];
      }),
      testLabels: testIdx.map(function (index) {
        return labels[index];
      }),
      trainLabels: trainIdx.map(function (index) {
        return labels[index];
      })
    };
  }

  exports.getFolds = getFolds;
  exports.getTrainTest = getTrainTest;
  exports.kFold = kFold;
  exports.leaveOneOut = leaveOneOut;
  exports.leavePOut = leavePOut;
  exports.sampleAClass = sampleAClass;

  Object.defineProperty(exports, '__esModule', { value: true });

}));
//# sourceMappingURL=ml-cross-validation.js.map
