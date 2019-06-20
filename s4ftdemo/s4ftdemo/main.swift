//
//  main.swift
//  s4ftdemo
//
//  Created by Angel Genov on 2019-06-16.
//  Copyright © 2019 Angel Genov. All rights reserved.
//

import Foundation
import TensorFlow

let epoch = 5000

//
// The Model
//
struct Model: Layer {
    var f1 = Dense<Float>(inputSize: 2, outputSize: 4, activation: relu)
    var f2 = Dense<Float>(inputSize: 4, outputSize: 4, activation: relu)
    var f3 = Dense<Float>(inputSize: 4, outputSize: 2)
    
    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: f1, f2, f3)
    }
}


//
// Training
//
let trainData = loadData(fileName: "/Users/agenov/Works/s4tf/s4ftdemo/data/set2.csv")

let x = Tensor<Float>(trainData.input)
let y = Tensor<Int32>(trainData.output)

var model = Model()
let opt = SGD(for: model)

    for i in 1...epoch {
        print("Starting training step \(i)")
        let (loss, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(x)
            return softmaxCrossEntropy(logits: logits, labels: y)
        }
        print("Loss: \(loss)")
        opt.update(&model.allDifferentiableVariables, along: grads)
    }


//
// Prediction (Inference)
//

let points: Tensor<Float> = [[-1.2411994293498834,-0.39836921369930917],
                             [0.39782214528725307,-0.947729019745054],
                             [-0.31584273105937305,-3.6766448944661083],
                             [-2.0534938741062176,3.000162533814112]]

var pred = model(points)
for i in 0..<pred.shape[0] {
    let classIdx = pred[i].argmax().scalar!
    print("Example \(i) prediction: \(Int(classIdx)) (\(softmax(pred[i])))")
}


//
// Evaluation
//

let testData = loadData(fileName: "/Users/agenov/Works/s4tf/s4ftdemo/data/set1.csv")
pred = model(Tensor<Float>(testData.input))
var res = Array<Int32>()

for i in 0..<pred.shape[0] {
    let classIdx = pred[i].argmax().scalar!
    res.append(classIdx)
}

if let correct = evaluate(predicted: res, real: testData.output) {
    let precision =  (Float(correct) / Float(res.count)) * 100
    print("\(correct) correct from \(res.count), precision is \(precision) %")
}
