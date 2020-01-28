'use strict';

const Dummy = require('./DummyClassifier');
const CV = require('..');

var LOO = require('./data/LOO-CV');
var LPO = require('./data/LPO-CV');
var KF = require('./data/KF-CV');


describe('basic', function () {
    it('basic leave-one-out cross-validation', function () {
        for (let i = 0; i < LOO.length; i++) {
            var CM = CV.leaveOneOut(Dummy, LOO[i].features, LOO[i].labels, LOO[i].classifierOptions);
            CM.getMatrix().should.deepEqual(LOO[i].result.matrix);
            CM.getLabels().should.deepEqual(LOO[i].result.labels);
        }
    });

    it('basic leave-p-out cross-validation', function () {
        for (let i = 0; i < LPO.length; i++) {
            var CM = CV.leavePOut(Dummy, LPO[i].features, LPO[i].labels, LPO[i].classifierOptions, LPO[i].p);
            CM.getMatrix().should.deepEqual(LPO[i].result.matrix);
            CM.getLabels().should.deepEqual(LPO[i].result.labels);
        }
    });

    it('basic k-fold cross-validation', function () {
        for (let i = 0; i < KF.length; i++) {
            var CM = CV.kFold(Dummy, KF[i].features, KF[i].labels, KF[i].classifierOptions, KF[i].k);
            CM.getMatrix().should.deepEqual(KF[i].result.matrix);
            CM.getLabels().should.deepEqual(KF[i].result.labels);
        }
    });
});

describe('with a callback', function () {
    it('basic leave-on-out cross-validation with callback', function () {
        for (let i = 0; i < LOO.length; i++) {
            var CM = CV.leaveOneOut(LOO[i].features, LOO[i].labels, function (trainFeatures, trainLabels, testFeatures) {
                const classifier = new Dummy(LOO[i].classifierOptions);
                classifier.train(trainFeatures, trainLabels);
                return classifier.predict(testFeatures);
            });
            CM.getMatrix().should.deepEqual(LOO[i].result.matrix);
            CM.getLabels().should.deepEqual(LOO[i].result.labels);
        }
    });

    it('basic leave-p-out cross-validation with callback', function () {
        for (let i = 0; i < LPO.length; i++) {
            var CM = CV.leavePOut(LPO[i].features, LPO[i].labels, LPO[i].p, function (trainFeatures, trainLabels, testFeatures) {
                const classifier = new Dummy(LPO[i].classifierOptions);
                classifier.train(trainFeatures, trainLabels);
                return classifier.predict(testFeatures);
            });
            CM.getMatrix().should.deepEqual(LPO[i].result.matrix);
            CM.getLabels().should.deepEqual(LPO[i].result.labels);
        }
    });

    it('basic k-fold cross-validation with callback', function () {
        for (let i = 0; i < KF.length; i++) {
            var CM = CV.kFold(KF[i].features, KF[i].labels, KF[i].k, function (trainFeatures, trainLabels, testFeatures) {
                const classifier = new Dummy(KF[i].classifierOptions);
                classifier.train(trainFeatures, trainLabels);
                return classifier.predict(testFeatures);
            });
            CM.getMatrix().should.deepEqual(KF[i].result.matrix);
            CM.getLabels().should.deepEqual(KF[i].result.labels);
        }
    });
});
