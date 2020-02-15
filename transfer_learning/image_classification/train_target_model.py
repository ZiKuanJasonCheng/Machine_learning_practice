import transfer_img_classification

'''
Transfer learning for image classification
Transfer from a dog recognition model to a cat recognition model
Run this file to train a target model
'''

if __name__ == "__main__":
  img_height = 32
  img_width = 32
  path = "./images/ILSVRC/Data/CLS-LOC/train/"

  # Load target data
  train_c1_list = ['n02123045', 'n02123159', 'n02123394']  # Cats dataset
  test_c1_list = train_c1_list
  train_c0_list = [('n01440764', 'n01443537']  # Other animals dataset
  test_c0_list = train_c0_list
  train_c1_interval = (0, 150)
  train_c0_interval = (1000, 1225)
  test_c1_interval = (150, 184)
  test_c0_interval = (1225, 1275)
  X_train, y_train, X_test, y_test = transfer_learning_github.read_data(path, train_c1_list, train_c1_interval, train_c0_list, train_c0_interval, test_c1_list,
                                               test_c1_interval, test_c0_list, test_c0_interval)

  load_ckpt_meta = './source_model/202002142042_model_epoch500.ckpt'  # Model checkpoint file path
  # The following settings are: initialize the weights of 2 conv filters from trained source model and freeze them during back propagation.
  dict_config = {'ckpt': load_ckpt_meta, 'init_scope': 'filters[12]', 'trainable_scope': 'filters3|dense[123]'}
  transfer_learning_github.train_model(X_train, y_train, X_test, y_test, img_height, img_width, source=False, )
