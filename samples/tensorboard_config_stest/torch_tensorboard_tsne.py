#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-11-2020
           """

import csv
import os
import re

import numpy
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

if __name__ == "__main__":

    def main(im_path="images"):
        def get_vector(input_image):
            image = input_image.convert(
                "RGB"
            )  # in case input image is not in RGB format
            img_t = transform(image)
            batch_t = torch.unsqueeze(img_t, 0)
            my_embedding = torch.zeros([1, 512, 1, 1])

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = layer.register_forward_hook(copy_data)
            model(batch_t)
            h.remove()
            return my_embedding.squeeze().cpu().numpy()

        model = models.resnet18(pretrained=True)
        layer = model._modules.get("avgpool")
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        im_names = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(im_path)
            for name in files
            if name.endswith(".jpg")
        ]
        existing_images_df = pd.DataFrame(
            [re.findall(r"[\w']+", im_name)[1:3] for im_name in im_names],
            columns=["cat_id", "pid"],
        )
        existing_images_df["impath"] = im_names
        vecs = [
            list(get_vector(Image.open(impath)))
            for _, pid, impath in existing_images_df.values
        ]

        with open("vis/feature_vecs.tsv", "w") as fw:
            csv_writer = csv.writer(fw, delimiter="\t")
            csv_writer.writerows(vecs)

        images = [
            Image.open(filename).resize((300, 300))
            for filename in existing_images_df["impath"]
        ]
        image_width, image_height = images[0].size
        one_square_size = int(numpy.ceil(numpy.sqrt(len(images))))
        master_width = image_width * one_square_size
        master_height = image_height * one_square_size

        spriteimage = Image.new(
            mode="RGBA", size=(master_width, master_height), color=(0, 0, 0, 0)
        )  # fully transparent

        for count, image in enumerate(images):
            div, mod = divmod(count, one_square_size)
            h_loc = image_width * div
            w_loc = image_width * mod
            spriteimage.paste(image, (w_loc, h_loc))

        spriteimage.convert("RGB").save("sprite.jpg", transparency=0)

        metadata = existing_images_df[["cat_id", "pid"]]
        metadata.to_csv("vis/metadata.tsv", sep="\t", index=False)

    main()
