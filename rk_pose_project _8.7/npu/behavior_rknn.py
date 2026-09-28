"""
============================================================
Behavior TCN RKNN Inference
RK3588 Runtime Version

Input:
    (16,46)

Output:
    label, confidence

Model:
    behavior_tcn.rknn

============================================================
"""


import numpy as np


try:

    from rknnlite.api import RKNNLite

except ImportError:

    raise ImportError(
        "Please install rknnlite"
    )



class BehaviorRKNN:


    def __init__(
        self,
        model_path
    ):


        print("[Behavior] load model")


        self.model_path = model_path



        # ==========================
        # class mapping
        # MUST match training
        # ==========================

        self.class_names = [

            "standing",

            "walking",

            "squat",

            "bend",

            "fall_down"

        ]



        # ==========================
        # RKNN
        # ==========================


        self.rknn = RKNNLite()



        ret = self.rknn.load_rknn(
            model_path
        )


        if ret != 0:

            raise RuntimeError(
                "load behavior model failed"
            )



        ret = self.rknn.init_runtime()


        if ret != 0:

            raise RuntimeError(
                "init behavior runtime failed"
            )


        print(
            "[Behavior] ready"
        )



    # ==================================================
    # Softmax
    # ==================================================

    def softmax(
        self,
        x
    ):


        x = x - np.max(x)


        exp = np.exp(x)


        return exp / np.sum(exp)



    # ==================================================
    # Forward
    # ==================================================

    def forward(
        self,
        seq
    ):


        """
        Input:

            seq:
                numpy.ndarray

                shape:
                (16,46)


        Return:

            label:
                str

            confidence:
                float

        """


        try:


            # ==========================
            # check input
            # ==========================


            data = np.asarray(
                seq,
                dtype=np.float32
            )


            if data.shape != (16,46):

                raise ValueError(
                    f"invalid input shape {data.shape}"
                )



            # ==========================
            # add batch
            # ==========================


            input_data = np.expand_dims(
                data,
                axis=0
            )


            # (1,16,46)



            # ==========================
            # RKNN inference
            # ==========================


            outputs = self.rknn.inference(

                inputs=[

                    input_data

                ]

            )



            if outputs is None:

                return (
                    "unknown",
                    0.0
                )



            logits = outputs[0]



            # remove batch

            logits = np.squeeze(
                logits
            )



            # ==========================
            # probability
            # ==========================


            prob = self.softmax(
                logits
            )



            pred_id = int(
                np.argmax(prob)
            )


            confidence = float(
                prob[pred_id]
            )



            label = self.class_names[
                pred_id
            ]



            return (

                label,

                confidence

            )



        except Exception as e:


            print(
                "[Behavior ERROR]",
                e
            )


            return (

                "unknown",

                0.0

            )



    # ==================================================
    # Release
    # ==================================================

    def release(self):


        try:

            self.rknn.release()

            print(
                "[Behavior] released"
            )


        except:

            pass
