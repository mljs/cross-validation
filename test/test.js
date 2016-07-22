'use strict';

const Dummy = require('./DummyClassifier');
const CV = require('..');

describe('basic', function () {
    it('basic leave-one-out cross-validation', function () {
        var LOO = require('./data/LOO-CV');
        for (let i = 0; i < LOO.length; i++) {
            var CM = CV.leaveOneOut(Dummy, LOO[i].features, LOO[i].labels, LOO[i].classifierOptions);
            CM.matrix.should.deepEqual(LOO[i].result.matrix);
            CM.labels.should.deepEqual(LOO[i].result.labels);
        }
    });
    
    it('basic leave-p-out cross-validation', function () {
        var LPO = require('./data/LPO-CV');
        for (let i = 0; i < LPO.length; i++) {
            var CM = CV.leavePOut(Dummy, LPO[i].features, LPO[i].labels, LPO[i].classifierOptions, LPO[i].p);
            CM.matrix.should.deepEqual(LPO[i].result.matrix);
            CM.labels.should.deepEqual(LPO[i].result.labels);
        }
    });

    it('basic k-fold cross-validation', function () {
        var KF = require('./data/KF-CV');
        for (let i = 0; i < KF.length; i++) {
            var CM = CV.kFold(Dummy, KF[i].features, KF[i].labels, KF[i].classifierOptions, KF[i].k);
            CM.matrix.should.deepEqual(KF[i].result.matrix);
            CM.labels.should.deepEqual(KF[i].result.labels);
        }
    });
});
