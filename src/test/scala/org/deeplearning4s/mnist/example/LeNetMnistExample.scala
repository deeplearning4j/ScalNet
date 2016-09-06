/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4s.mnist.example

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4s.layers.convolutional.Convolution2D
import org.deeplearning4s.layers.pooling.MaxPooling2D
import org.deeplearning4s.layers.{Dense, DenseOutput}
import org.deeplearning4s.models.{Sequential}
import org.deeplearning4s.optimizers.SGD
import org.deeplearning4s.regularizers.l2
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

/**
  * Two-layer MLP for MNIST.
  *
  * @author David Kale
  */
object LeNetMnistExample extends App {
  private val log: Logger = LoggerFactory.getLogger("yay")

  private val nbRows: Int = 28
  private val nbColumns: Int = 28
  private val nbChannels: Int = 1
  private val nbOutput: Int = 10
  private val batchSize: Int = 64
  private val nbEpochs: Int = 1
  private val rngSeed: Int = 123
  private val weightDecay: Double = 0.0005
  private val momentum: Double = 0.9
  private val learningRate: Double = 0.01

  private val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, 12345)
  private val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, 12345)

  log.info("Build model....")
  private val model: Sequential = new Sequential(inputShape = List(nbRows, nbColumns, nbChannels), rngSeed = rngSeed)
  model.add(new Convolution2D(20, nChannels = nbChannels, kernelSize = List(5, 5), stride = List(1, 1),
                              weightInit = WeightInit.XAVIER, activation = "identity", regularizer = l2(weightDecay)))
  model.add(new MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
  model.add(new Convolution2D(50, kernelSize = List(5, 5), stride = List(1, 1),
                              weightInit = WeightInit.XAVIER, activation = "identity", regularizer = l2(weightDecay)))
  model.add(new MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
  model.add(new Dense(500, nbRows*nbColumns, weightInit = WeightInit.XAVIER, activation = "relu", regularizer = l2(weightDecay)))
  model.add(new DenseOutput(nbOutput, weightInit = WeightInit.XAVIER, activation = "softmax", lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD))
  model.compile(optimizer = SGD(learningRate, momentum = momentum, nesterov = true))

  log.info("Train model....")
  model.fit(mnistTrain, nbEpoch = nbEpochs, List(new ScoreIterationListener(1)))

  log.info("Evaluate model....")
  val evaluator: Evaluation = new Evaluation(nbOutput)
  while(mnistTest.hasNext){
    val next: DataSet = mnistTest.next()
    val output: INDArray = model.predict(next)
    evaluator.eval(next.getLabels, output)
  }
  log.info(evaluator.stats())
  log.info("****************Example finished********************")
}
