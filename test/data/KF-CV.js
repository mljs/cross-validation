'use strict';

'use strict';

module.exports = [
    {
        features: [-10, 0, 5],
        labels: [-1, 0, 1],
        k: 3,
        result: {
            accuracy: 1,
            confusionMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            labels: [-1, 0, 1],
            nbPrediction: 3
        }
    },
    {
        features: [-3, -10, -2, 5],
        labels: [1, 1, -1, 1],
        k: 2,
        result: {
            accuracy: 1 / 2,
            confusionMatrix: [[1, 2], [0, 1]],
            labels: [1, -1],
            nbPrediction: 4
        }
    }
];