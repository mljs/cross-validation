'use strict';

const CV = module.exports = {};
const combinations = require('ml-combinations');

// Returns a confusion matrix
CV.leaveOneOut = function (Classifier, features, labels, classifierOptions) {
    return CV.leavePOut(Classifier, features, labels, classifierOptions, 1);
};


CV.leavePOut = function (Classifier, features, labels, classifierOptions, p) {
    check(features, labels);
    const distinct = getDistinct(labels);
    const confusionMatrix = initMatrix(distinct.length, distinct.length);
    var correct = 0, total = 0;
    var i, N = features.length;
    var gen = combinations(p, N);
    var allIdx = new Array(N);
    for (i = 0; i < N; i++) {
        allIdx[i] = i;
    }
    for (const testIdx of gen) {
        var trainIdx = allIdx.slice();
        
        for (i = testIdx.length - 1; i >= 0; i--) {
            trainIdx.splice(testIdx[i], 1);
        }

        var testFeatures = testIdx.map(function (index) {
            return features[index];
        });
        var trainFeatures = trainIdx.map(function (index) {
            return features[index];
        });
        var testLabels = testIdx.map(function (index) {
            return labels[index];
        });
        var trainLabels = trainIdx.map(function (index) {
            return labels[index];
        });

        var classifier = new Classifier(classifierOptions);
        classifier.train(trainFeatures, trainLabels);
        var predictedLabels = classifier.predict(testFeatures);
        for (i = 0; i < predictedLabels.length; i++) {
            total++;
            if (testLabels[i] === predictedLabels[i]) {
                correct++;
            }
            confusionMatrix[distinct.indexOf(testLabels[i])][distinct.indexOf(predictedLabels[i])]++;
        }
    }
    return {
        confusionMatrix,
        accuracy: correct / total,
        labels: distinct,
        nbPrediction: total
    };
};

function check(features, labels) {
    if (features.length !== labels.length) {
        throw new Error('features and labels should have the same length');
    }
}

function initMatrix(rows, columns) {
    return new Array(rows).fill(0).map(() => new Array(columns).fill(0));
}

function getDistinct(arr) {
    var s = new Set();
    for (let i = 0; i < arr.length; i++) {
        s.add(arr[i]);
    }
    return Array.from(s);
}