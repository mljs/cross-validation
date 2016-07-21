'use strict';

const Dummy = require('./DummyClassifier');
const CV = require('..');

describe('basic', function () {
    it('basic leave-one-out cross-validation', function () {
        var LOO = require('./data/LOO-CV');
        for(let i=0; i<LOO.length; i++) {
            var result = CV.leaveOneOut(Dummy, LOO[i].features, LOO[i].labels, LOO[i].classifierOptions);
            result.should.deepEqual(LOO[i].result);
        }
    });
    
    it('basic leave-p-out cross-validation', function () {
        var LPO = require('./data/LPO-CV');
        for(let i=0; i<LPO.length; i++) {
            var result = CV.leavePOut(Dummy, LPO[i].features, LPO[i].labels, LPO[i].classifierOptions, LPO[i].p);
            result.should.deepEqual(LPO[i].result);
        }
    });

    it('basic k-fold cross-validation', function () {
        var KF = require('./data/KF-CV');
        for(let i=0; i<KF.length; i++) {
            var result = CV.kFold(Dummy, KF[i].features, KF[i].labels, KF[i].classifierOptions, KF[i].k);
            result.should.deepEqual(KF[i].result);
        }
    });
});
