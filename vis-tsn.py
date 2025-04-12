from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
tsne = TSNE(n_components=2)
x = [[1,2,2],[2,2,2],[3,3,3]]
y = [1,0,2]  # 标签
x_tsne = tsne.fit_transform(x)
# plt.scatter(x_tsne[:,0], x_tsne[:,1], c=y)
# plt.show()
sem = F.to_pil_image(x_tsne)
sem.save('./test_tsn.jpg')