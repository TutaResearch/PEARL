# PEARL
Tensorflow implementation of "Personalized Federated Graph Learning on Non-IID Electronic Health Records".

## Dependencies
- python>=3.7
- tensorflow==2.3.0
- numpy
- scipy
- pandas

Write the following code in "Tf.keras.models" model:

```bash
def set_weights(self, weight):
    """set the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    with self.distribute_strategy.scope():
      return super(Model, self).set_weights(weight)
```
Easily find the place: In the same class as "PEARL_pretrain.get_weights()" in "FL_train.py".
The form of "set_weights" is the same as that of "get_weights".

## Usage

The parameters are configured in each python file.

```bash
python run_preprocess.py

python run_hyperbolic_embedding.py

python FL_train.py
```
