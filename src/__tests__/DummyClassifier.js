'use strict';

class Dummy {
    constructor() {
        this.threshold = 0;
    }

    train() {
        // train does nothing, this classifier is really dumb
    }

    predict(features) {
        if (!Array.isArray(features)) {
            features = [features];
        }
        var labels = new Array(features.length);

        for (let i = 0; i < features.length; i++) {
            if (features[i] === this.threshold) {
                labels[i] = 0;
            } else if (features[i] < this.threshold) {
                labels[i] = -1;
            } else {
                labels[i] = 1;
            }
        }
        return labels;
    }
}

module.exports = Dummy;
