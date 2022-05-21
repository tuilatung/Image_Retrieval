import PIL
from PIL import Image
from utils import *
import matplotlib.pyplot as plt
import pickle
import math

def query_image(img_query):
    model = get_extract_model()
    # Query image features extraction
    search_vector = extract_vector(model, img_query)

    vectors = pickle.load(open("./results/vectors.pkl","rb"))
    paths = pickle.load(open("./results/paths.pkl","rb"))

    # Distance from query's vector to all vector in dataset
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    K = 10 # Return top K image same as query image 
    ids = np.argsort(distance)[:K]

    nearest_image = [(paths[id], distance[id]) for id in ids]

    axes = []
    grid_size = int(math.sqrt(K))
    fig = plt.figure(figsize=(20,10))


    for id in range(K):
        draw_image = nearest_image[id]
        # axes.append(fig.add_subplot(grid_size, grid_size, id+1))
        axes.append(fig.add_subplot(5, 5, id+1))


        # axes[-1].set_title(draw_image[1])
        # axes[-1].set_title(draw_image[0].split('/')[-1] + f' ({draw_image[1]})')
        axes[-1].set_title(f'Top #{id+1}')

        plt.axis('off')
        plt.imshow(Image.open(draw_image[0].replace('/content', './dataset')))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_query_path = './dataset/img/23431.png'
    img_query = Image.open(img_query_path)
    plt.imshow(img_query)
    plt.show()
    query_image(img_query_path)