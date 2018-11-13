# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Main file for LSTM and dense layer example

Main file for LSTM example to be used with LSTM architecture in sample_architectures and config file config_lstm.json;
Also to be used with other LSTM example architectures and a dense layer architecture (see sample_architectures.py for
the different examples and descriptions);

"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

#
# Imports before spawning workers (do NOT import tensorflow or matplotlib here)
#
import sys
from collections import OrderedDict
import numpy as np
import progressbar

# Import TeLL
from TeLL.config import Config
from TeLL.datareaders import initialize_datareaders, DataLoader
from TeLL.utility.misc import AbortRun, check_kill_file
from TeLL.utility.timer import Timer
from TeLL.utility.workingdir import Workspace
if __name__ == "__main__":
    from TeLL.session import TeLLSession

    import tensorflow as tf
    from TeLL.regularization import regularize


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def update_step(loss, config, clip_gradient=1., scope='optimizer'):
    """Computation of gradients and application of weight updates

    Optimizer can be supplied via config file, e.g. as
    "optimizer_params": {"learning_rate": 1e-3},
    "optimizer": "'AdamOptimizer'"

    Parameters
    -------
    loss : tensor
        Tensor representing the
    config : config file
        Configuration file
    clip_gradient : (positive) float or False
        Clip gradient at +/- clip_gradient or don't clip gradient if clip_gradient=False

    Returns
    -------
    : tensor
        Application of gradients via optimizer
    """
    # Define weight update
    with tf.variable_scope(scope):
        trainables = tf.trainable_variables()
        # Set optimizer (one learning rate for all layers)
        optimizer = getattr(tf.train, config.optimizer)(**config.optimizer_params)

        # Calculate gradients
        gradients = tf.gradients(loss, trainables)
        # Clip all gradients
        if clip_gradient:
            gradients = [tf.clip_by_value(grad, -clip_gradient, clip_gradient) for grad in gradients]
        # Set and return weight update
        return optimizer.apply_gradients(zip(gradients, trainables))


def evaluate_on_validation_set(validationset, step: int, session, model, summary_writer, validation_summary,
                               val_loss, workspace: Workspace):
    """Convenience function for evaluating network on a validation set

    Parameters
    -------
    validationset : dataset reader
        Dataset reader for the validation set
    step : int
        Current step in training
    session : tf.session
        Tensorflow session to use
    model : network model
        Network model
    val_loss : tensor
        Tensor representing the validation loss computation

    Returns
    -------
    : float
        Loss averaged over validation set
    """
    loss = 0

    _pbw = ['Evaluating on validation set: ', progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=_pbw, maxval=validationset.n_mbs - 1, redirect_stdout=True).start()

    mb_validation = validationset.batch_loader()

    with Timer(verbose=True, name="Evaluate on Validation Set"):
        for mb_i, mb in enumerate(mb_validation):
            # Abort if indicated by file
            check_kill_file(workspace)

            val_summary, cur_loss = session.run([validation_summary, val_loss],
                                                feed_dict={model.X: mb['X'], model.y_: mb['y']})

            loss += cur_loss
            progress.update(mb_i)

            mb.clear()
            del mb

    progress.finish()

    avg_loss = loss / validationset.n_mbs

    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Validation Loss", simple_value=avg_loss)]),
                               step)

    print("Validation scores:\n\tstep {} validation loss {}".format(step, avg_loss))
    sys.stdout.flush()

    return avg_loss


def main(_):
    # ------------------------------------------------------------------------------------------------------------------
    # Setup training
    # ------------------------------------------------------------------------------------------------------------------

    # Initialize config, parses command line and reads specified file; also supports overriding of values from cmd
    config = Config()

    random_seed = config.get_value('random_seed', 12345)
    np.random.seed(random_seed)  # not threadsafe, use rnd_gen object where possible
    rnd_gen = np.random.RandomState(seed=random_seed)

    # Load datasets for trainingset
    with Timer(name="Loading Data"):
        readers = initialize_datareaders(config, required=("train", "val"))
        trainingset = DataLoader(readers["train"], batchsize=config.batchsize)
        #validationset = DataLoader(readers["val"], batchsize=config.batchsize)

        # Initialize TeLL session
    tell = TeLLSession(config=config, model_params={"input_shape": [300]})

    # Get some members from the session for easier usage
    session = tell.tf_session

    model = tell.model
    workspace, config = tell.workspace, tell.config



    # Initialize Tensorflow variables
    global_step = tell.initialize_tf_variables().global_step

    sys.stdout.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------------------------------------------------------

    try:
        epoch = int(global_step / trainingset.n_mbs)
        epochs = range(epoch, config.n_epochs)

        #
        # Loop through epochs
        #
        print("Starting training")

        for ep in epochs:
            print("Starting training epoch: {}".format(ep))
            # Initialize variables for over-all loss per epoch
            train_loss = 0

            # Load one minibatch at a time and perform a training step
            t_mb = Timer(name="Load Minibatch")
            mb_training = trainingset.batch_loader(rnd_gen=rnd_gen)

            #
            # Loop through minibatches
            #

            for mb_i, mb in enumerate(mb_training):
                sys.stdout.flush()
                #Print minibatch load time
                t_mb.print()

                # Abort if indicated by file
                check_kill_file(workspace)

                #
                # Calculate scores on validation set
                #
                if global_step % config.score_at == 0:
                    print("Starting scoring on validation set...")


                # Get new sample
                training_sample = np.ones(shape=(1,np.random.randint(low=20,high=100),300))
                #
                # Perform weight update
                #
                with Timer(name="Weight Update"):

                    #
                    # Set placeholder values
                    #
                    placeholder_values = OrderedDict(
                        input_placeholder=training_sample,
                        sequence_length_placeholder = training_sample.shape[1]
                    )
                    feed_dict = dict(((model.placeholders[k], placeholder_values[k]) for k in placeholder_values.keys()))

                    #
                    # Decide which tensors to compute
                    #
                    data_keys = ['lstm_internals_enc', 'lstm_internals_dec', 'lstm_h_enc',
                                 'lstm_h_dec', 'loss' , 'loss_last_time_prediction','loss_last_time_prediction', 'reg_loss']
                    data_tensors = [model.data_tensors[k] for k in data_keys]

                    operation_keys = ['ae_update']
                    operation_tensors = [model.operation_tensors[k] for k in operation_keys]

                    summary_keys = ['all_summaries']
                    summary_tensors = [model.summaries[k] for k in summary_keys]

                    #
                    # Run graph and re-associate return values with keys in dictionary
                    #
                    ret = session.run(data_tensors + summary_tensors + operation_tensors, feed_dict)

                    data_keys = ['loss']
                    data_tensors = [model.data_tensors[k] for k in data_keys]
                    session.run(model.data_tensors['loss'] , feed_dict)
                    session.run(model.data_tensors['latent_space'], feed_dict)

                    ret_dict = OrderedDict(((k, ret[i]) for i, k in enumerate(data_keys)))
                    del ret[:len(data_keys)]
                    ret_dict.update(OrderedDict(((k, ret[i]) for i, k in enumerate(summary_keys))))






                # Print some status info
                #print("ep {} mb {} loss {} (avg. loss {})".format(ep, mb_i, cur_loss, train_loss / (mb_i + 1)))

                # Reset timer
                #t_mb = Timer(name="Load Minibatch")

                # Free the memory allocated for the minibatch data
                #mb.clear()
                #del mb

                global_step += 1

            #
            # Calculate scores on validation set after training is done
            #

            # Perform scoring on validation set
            print("Starting scoring on validation set...")


            tell.save_checkpoint(global_step=global_step)

            # Abort if indicated by file
            check_kill_file(workspace)

    except AbortRun:
        print("Detected kill file, aborting...")

    finally:
        tell.close(save_checkpoint=True, global_step=global_step)


if __name__ == "__main__":
    tf.app.run()
