from preprocessing.preprocessing_data import prepare_train_batches
from models.u_net_model import create_model


def train_model(train_batches, epochs=10, steps_per_epoch=1000):
    model = create_model()
    model.fit(train_batches, epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model


def train_and_save_model():
    train_batches = prepare_train_batches()
    model = train_model(train_batches)
    model.save_weights('./checkpoints/my_checkpoint')


train_and_save_model()
