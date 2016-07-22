'use strict';

module.exports = [
    {
        features: [-10, 0, 5],
        labels: [-1, 0, 1],
        p: 2,
        result: {
            accuracy: 1,
            confusionMatrix: [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            labels: [-1, 0, 1],
            nbPrediction: 6
        }
    },
    {
        features: [-3, -10, -2, 5],
        labels: [1, 1, -1, 1],
        p: 2,
        result: {
            accuracy: 1 / 2,
            confusionMatrix: [[3, 6], [0, 3]],
            labels: [1, -1],
            nbPrediction: 12
        }
    }
];
