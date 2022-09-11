from constants import EPOCHS, STEPS_PER_EPOCH
from preprocessing.preprocessing_data import prepare_train_and_validation_batches
from models.u_net_model import create_model


def train_model(train_batches, validation_batches, epochs=10, steps_per_epoch=1000):
    model = create_model()
    model.fit(train_batches,
              validation_data=validation_batches,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch)
    return model


def train_and_save_model():
    train_batches, validation_batches = prepare_train_and_validation_batches()
    model = train_model(train_batches, validation_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    model.save_weights('./checkpoints/my_checkpoint')


train_and_save_model()
