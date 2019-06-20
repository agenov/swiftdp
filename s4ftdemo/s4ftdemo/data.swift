//
//  data.swift
//  model1
//
//  Created by Angel Genov on 2019-06-16.
//  Copyright © 2019 Angel Genov. All rights reserved.
//

import Foundation
import TensorFlow

func loadData(fileName: String) -> (input: ShapedArray<Float>, output: Array<Int32>) {
    var points = Array<Float>()
    var labels = Array<Int32>()
    
    do {
        let data = try String(contentsOfFile: fileName)
        let lines = data.components(separatedBy: .newlines)
        for l in lines {
            let components = l.components(separatedBy: ",")
            points.append((components[0] as NSString).floatValue)
            points.append((components[1] as NSString).floatValue)
            let label = Int32((components[2] as NSString).intValue > 0 ? 1 : 0)
            labels.append(label)
        }
    } catch {
        print(error)
        exit(1)
    }
    
    let samplesCount = points.count / 2
    let input = ShapedArray<Float>(shape: [samplesCount, 2], scalars: points)
    return (input: input, output: labels)
}

func evaluate(predicted:Array<Int32>, real:Array<Int32>) -> Int? {
    guard predicted.count == real.count else {
        return nil
    }
    
    var correct = 0
    for (i,v) in predicted.enumerated() {
        if v == real[i] {
            correct += 1
        }
    }
    
    return correct
}
