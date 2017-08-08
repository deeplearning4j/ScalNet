package org.deeplearning4j.scalnet.optimizers

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.nd4j.linalg.learning.config._

/**
  * Optimizers for neural nets.
  *
  * @author David Kale
  */
sealed class Optimizer(val optimizationAlgorithm: OptimizationAlgorithm, val lr: Double = 1e-1)
case class SGD(override val lr: Double = Sgd.DEFAULT_SGD_LR)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
case class RMSPROP(override val lr: Double = RmsProp.DEFAULT_RMSPROP_LEARNING_RATE, rmsdecay: Double = RmsProp.DEFAULT_RMSPROP_RMSDECAY, epsilon: Double = RmsProp.DEFAULT_RMSPROP_EPSILON)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
case class NESTEROVS(override val lr: Double = Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE, momentum: Double = Nesterovs.DEFAULT_NESTEROV_MOMENTUM)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
case class ADAM(override val lr: Double = Adam.DEFAULT_ADAM_LEARNING_RATE, meandecay: Double = Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY, vardecay: Double = Adam.DEFAULT_ADAM_BETA2_VAR_DECAY, epsilon: Double = Adam.DEFAULT_ADAM_EPSILON)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
case class ADADELTA(rho: Double = AdaDelta.DEFAULT_ADADELTA_RHO, epsilon: Double = AdaDelta.DEFAULT_ADADELTA_EPSILON)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
case class ADAGRAD(override val lr: Double = AdaGrad.DEFAULT_ADAGRAD_LEARNING_RATE, epsilon: Double = AdaGrad.DEFAULT_ADAGRAD_EPSILON)
  extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
object NONE extends Optimizer(optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
