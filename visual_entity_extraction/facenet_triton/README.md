# Facenet Inference 
Dockerised Facenet Inference Model

### Data
Inside the data folder, there are some images and embeddings for testing.
* Folder `data/2576` contains 25 photos of id 2576
* Folder `data/pt_emb` contains embeddings generated for 1000 individuals (excluding 2576). 
* The other images inside `data` folder are for inference testing. 

#### Note:
All individuals must be tagged with an numerical id. For generation of embedding, the folder name will serve as the numerical id (as with `data/2576`). If identity of individuals are required, you may use a `.json` file or something to save the identity mapping.

### face_id_api (FastAPI)
* Inside `src/config.yaml`, indicate the path to the embedding folders. (Even if there is no embedding yet, point it to an empty folder, and populated embeddings will be stored in that folder as well.)
* Select the model served on Triton as well (i.e. `triton/models/<model to serve>`)

#### API Endpoints
| Method | Endpoint | Description  |
|---|---|---|
| POST | /infer | Takes in an object {"image": im_b64} and predicts the id of the face based on existing list of embeddings <br> Returns the id, confidence and the corresponding bounding box. |
| POST | /generate | Takes in an object {"id": int, "images": List[Dict['image': im_b64]]}, generates an embedding and saves it locally. |


### mtcnn (Flask)
It was originally intended for MTCNN models to be served on Triton as well, however, due to dependency issues (NumPy), and conditional statements, this model is not suitable for conversion to TorchScript. No further configurations required here. 

### triton (NVIDIA Triton Inference Server)
With reference to [Triton's Inference Server Repository](https://github.com/triton-inference-server/server), files need to be organised in the following layout:

```
.              
├── models
│   ├── <model1>
|   |   ├── <version1>
|   |   |   └── model.pt
|   |   ├── <version2>
|   |   |   └── model.pt
|   |   └── config.pbtxt
│   └── <model2>
|       ├── <version1>
|       |   └── model.pt
|       ├── <version2>
|       |   └── model.pt
|       └── config.pbtxt
└── Dockerfile
```

An example of such repository is shown in the `Triton` folder. 

Note: `.pt` file MUST be renamed as `model.pt`

#### Dockerfile
- Used to build the repository into a docker image. Key thing to adjust here is the version of `tritonserver` base docker image, as necessary.

#### model.pt 
- Copy the `model.pt` generated in the previous step into the appropriate path as shown in the layout above. 

#### config.pbtxt

- Each model requires a `config.pbtxt` file that, most importantly, describes the input and output shape of the model. The basic `config.pbtxt` file for PyTorch is in `Triton\models\resnet` and `Triton\models\bert` folder. 
- `data_type`: For each input and output, the `data_type` needs to specified. The valid `data_type` and their PyTorch equivalent can be found in [Triton's Server Model Configuration Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes)
- `dims`: Specify the dimensions of a single input example. In cases where the dimension varies (e.g. in the case of width and height of images), `-1` will be listed for those dimension (Example shown in `Triton\models\bert\config.pbtxt`, where texts have varying sequence length)

The example in this repository describes the very basic Triton repository for PyTorch. All other specifications and configurations of `config.pbtxt` (e.g. `version_policy`, `max_batch_size` etc) can be found in the [documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)


### Model
#### Prepare a PyTorch model for inference with Triton
For PyTorch model to be used with Triton, it first needs to be converted into Torchscript model (by tracing).
Notes:
- Input of model during tracing should be of type `torch.Tensor`. Other input types may cause incorrect tracing, or might not work at all. 
- If custom model consist of conditional logics that affects the final output, then the portion must be isolated and scripted instead. Refer to [PyTorch's Mixing Tracing and Scripting](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting) for more details. 
    - Scripting would not be necessary if the model always go through a single logic flow only when deployed.

##### Quick Start
1. Install:

    ```bash
    # With pip:
    pip install facenet-pytorch
    
    # or clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    
    # or use a docker container (see https://github.com/timesler/docker-jupyter-dl-gpu):
    docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython
    ```
    
2. Run `trace_model.py` to convert the VGGFace2 pretrained Facenet model into a Torchscript model. The script automatically downloads the VGGFace2 pretrained Facenet model. However, you can manually download the VGGFace2 pretrained Facenet Model state_dict's from [Tim Esler's Facenet PyTorch Repository](https://github.com/timesler/facenet-pytorch) and load it into the script if required. The traced model will be saved as `model.pt`.

3. Move the model into the `triton/models/facenet/1/` folder.

4. client.ipynb has the examples on how to make inference and generate new embedding.

    You may try to infer id_is_2576.jpg before generating 2576's embedding, it will return a wrong prediction with relatively low confidence.
    You can then proceed to generate the embedding for 2576, then make another inference afterwards. You should obtain a much higher confidence with the correct id prediction.

5. To save and query embeddings to local, set `dataset = EmbeddingDataset(EMBEDDING_PATH, local=True)` and `uploader = Uploader(EMBEDDING_PATH, local=True)` in `face_id_api/main.py`. By default, embeddings are saved to and inferred from the ElasticSearch specified in `config.yaml`.
    

### Configurations
Refer to `docker-compose.yml` for the volumes mounting

### Usage
1) Run `docker-compose up` to spin up all the services
2) When the triton server is successfully served, you should see the following:
```
+----------+---------+--------+
| Model    | Version | Status |
+----------+---------+--------+
| facenet  | 1       | READY  |
+----------+---------+--------+

```
3) `client.ipynb` has the examples on how to make inference and generate new embedding. 
    * You may try to infer `id_is_2576.jpg` before generating 2576's embedding, it will return a wrong prediction with relatively low confidence. 
    * You can then proceed to generate the embedding for 2576, then make another inference afterwards. You should obtain a much higher confidence with the correct id prediction.

#### Known Constraints:
* Images with transparent background (typically `.png`) throws error when being fed into MTCNN
* In order to delete embedding, server have to be stopped, manually delete file, then restart. No delete function has been implemented
* If there are too many faces found, then error will be thrown (When no. of faces detected > triton batch size indicated in `config.pbtxt`)

#### Possible Areas to Explore:
* Deleting embeddings by ID
* Deleting all embeddings
* Updating current embeddings (without images which generated current embedding). Currently, any updates would overwrite existing embeddings.
* Merging saved embeddings (i.e. 2 saved embeddings identified to be from the same individual)
