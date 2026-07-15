import os
import torch
import joblib
import numpy as np

from decision_models.imitation_net import ImitationNet


TREE_PATH = "./decision_tree/tree_model.pkl"
MODEL_DIR = "./decision_models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


QP_VALUES = [0, 1, 2, 3, 4]
SKIP_VALUES = [0, 1, 2, 3, 4]
RE_VALUES = [0, 1, 2, 3, 4]


HISTORY_LENGTH = 8


class OnlineSwitcher:

    def __init__(self):

        self.tree = joblib.load(
            TREE_PATH
        )

        self.models = {}

        self.load_models()


        self.history_action = np.zeros(
            (
                HISTORY_LENGTH,
                3
            ),
            dtype=np.int64
        )


    def load_models(self):

        for file in os.listdir(MODEL_DIR):

            if file.startswith("model_") and file.endswith(".pth"):

                model_id = int(
                    file.split("_")[1].split(".")[0]
                )


                model = ImitationNet()

                model.load_state_dict(
                    torch.load(
                        os.path.join(
                            MODEL_DIR,
                            file
                        ),
                        map_location=DEVICE
                    )
                )


                model.to(DEVICE)

                model.eval()


                self.models[model_id] = model



    def select_model(
        self,
        degradation_feature
    ):


        feature = np.asarray(
            degradation_feature
        )


        if feature.ndim == 1:

            feature = feature.reshape(
                1,
                -1
            )


        model_id = self.tree.predict(
            feature
        )[0]


        return int(model_id)



    @torch.no_grad()
    def inference(
        self,
        model_id,
        bandwidth,
        p1_feature,
        diff_feature
    ):


        model = self.models[
            model_id
        ]


        bandwidth = torch.tensor(
            bandwidth,
            dtype=torch.float32,
            device=DEVICE
        )


        history_action = torch.tensor(
            self.history_action,
            dtype=torch.long,
            device=DEVICE
        )


        p1_feature = torch.tensor(
            p1_feature,
            dtype=torch.float32,
            device=DEVICE
        )


        diff_feature = torch.tensor(
            diff_feature,
            dtype=torch.float32,
            device=DEVICE
        )


        bandwidth = bandwidth.reshape(
            1,
            -1
        )


        history_action = history_action.unsqueeze(
            0
        )


        p1_feature = p1_feature.unsqueeze(
            0
        )


        diff_feature = diff_feature.unsqueeze(
            0
        )



        output = model(
            bandwidth,
            history_action,
            p1_feature,
            diff_feature
        )


        qp_id = torch.argmax(
            output["qp"],
            dim=-1
        ).item()


        skip_id = torch.argmax(
            output["skip"],
            dim=-1
        ).item()


        re_id = torch.argmax(
            output["re"],
            dim=-1
        ).item()



        action = np.array(
            [
                qp_id,
                skip_id,
                re_id
            ]
        )


        self.history_action[:-1] = (
            self.history_action[1:]
        )


        self.history_action[-1] = action



        return {

            "model_id": model_id,

            "qp": QP_VALUES[qp_id],

            "skip": SKIP_VALUES[skip_id],

            "re": RE_VALUES[re_id]

        }




def online_switching():


    switcher = OnlineSwitcher()


    segment_num = 10


    for segment_id in range(segment_num):


        bandwidth = np.random.uniform(
            1,
            5
        )


        degradation_feature = np.random.randn(
            16
        )


        p1_feature = np.random.randn(
            8,
            40
        )


        diff_feature = np.random.randn(
            8,
            40
        )



        model_id = switcher.select_model(
            degradation_feature
        )


        result = switcher.inference(
            model_id,
            bandwidth,
            p1_feature,
            diff_feature
        )


        print(
            "Segment:",
            segment_id,
            "Model ID:",
            result["model_id"],
            "QP:",
            result["qp"],
            "SKIP:",
            result["skip"],
            "RE:",
            result["re"]
        )



if __name__ == "__main__":
    online_switching()