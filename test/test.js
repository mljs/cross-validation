'use strict';

const Dummy = require('./DummyClassifier');
const CV = require('..');

describe('basic', function () {
    it('basic', function () {
        var basic = require('./data/basic');
        for(let i=0; i<basic.length; i++) {
            var result = CV.leaveOneOut(Dummy, basic[i].features, basic[i].labels, basic[i].classifierOptions);
            result.should.deepEqual(basic[i].result);
        }
    });
});