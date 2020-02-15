import transfer_img_classification

'''
Transfer learning for image classification
Transfer from a dog recognition model to a cat recognition model
Run this file to train a source model
'''

if __name__ == "__main__":
  img_height = 32
  img_width = 32
  path = './images/ILSVRC/Data/CLS-LOC/train/'  # Directory path of image dataset

  # Load source data
  # Image data are from ImageNet
  train_c1_list = test_c1_list = ['n02111277', 'n02111129', 'n02110958', 'n02110806', 'n02110627', 'n02110341']  # Dogs dataset
  train_c0_list = test_c0_list = ['n01440764', 'n01443537']  # Other animals dataset
  train_c1_interval = (0, 300)
  train_c0_interval = (0, 900)
  test_c1_interval = (300, 335)
  test_c0_interval = (900, 1000)
  X_train, y_train, X_test, y_test = transfer_learning_github.read_data(path, train_c1_list, train_c1_interval, train_c0_list, train_c0_interval, test_c1_list,
                                               test_c1_interval, test_c0_list, test_c0_interval, img_height, img_width)

  # Build a model(graph)
  #graph, saver = transfer_learning_github.build_graph()
  # Train a model
  transfer_learning_github.train_model(X_train, y_train, X_test, y_test, img_height, img_width)
