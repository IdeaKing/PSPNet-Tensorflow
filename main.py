import tensorflow as tf

import train, dataset
from models import PSPNet
from losses import loss
from utils import arg_parser

def main(args):
    if args.debug:
        tf.config.run_functions_eagerly(True)
    # Set the precision
    if args.precision != ("mixed_float16" or "float32"):
        ValueError(f"{args.precision} is not a precision type.")
    tf.keras.mixed_precision.set_global_policy(args.precision)

    # Create the dataset
    file_names = dataset.load_data(dataset_path=args.dataset_path)
    dataset_creater = dataset.Dataset(file_names=file_names,
                                      dataset_path=args.dataset_path,
                                      batch_size=args.batch_size,
                                      shuffle_size=args.shuffle_size,
                                      images_dir=args.images_dir,
                                      labels_dir=args.labels_dir,
                                      image_dims=args.image_dims,
                                      augment_ds=args.augment_ds)
    labeled_ds = dataset_creater()

    # Define training parameters
    total_steps = int((len(file_names) / args.batch_size) * args.epochs)
    num_classes = 3

    # Create the model
    model = PSPNet.get_pspnet(name=args.model,
                              num_classes=num_classes,
                              input_shape=args.image_dims)

    
    # Define the loss function
    losses = loss.CategoricalFocalLoss()

    # Define the optimizer
    if args.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                            momentum=args.optimizer_momentum)
    elif args.optimizer == "ADAM":
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    else:
        raise ValueError(f"{args.optimizer} is not an available optimizer")
    
    # Define the training function
    train_func = train.Train(training_dir=args.training_dir,
                             epochs=args.epochs,
                             total_steps=total_steps,
                             input_shape=args.image_dims,
                             precision=args.precision,
                             max_checkpoints=args.max_checkpoints,
                             checkpoint_frequency=args.checkpoint_frequency,
                             save_model_frequency=args.save_model_frequency,
                             print_loss=args.print_loss,
                             log_every_step=args.log_every_step,
                             from_checkpoint=args.from_checkpoint)

    # Train the model
    train_func.supervised(dataset=labeled_ds,
                          model=model,
                          losses=losses,
                          optimizer=optimizer)

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    args = arg_parser.args
    main(args)