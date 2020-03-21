import numpy as np
import pickle
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
import os
from google.protobuf import text_format
import tensorflow as tf
import zlib

img_root_folder = "/data/shared/ConceptualCaptions/menekse_images/"
names_img_map = "Train_GCC-training.tsv"


with open(names_img_map) as f:
    data = f.readlines()

names_data = []
for i,line in enumerate(data):
    try:
        names_data.append(str(i) + "_" + str(zlib.crc32(line.split("\t")[-1].rstrip().encode('utf-8')) & 0xffffffff))
    except Exception as e:
        print(e)
        continue
# load the pipeline structure from the config file
with open('models/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_oid_v4.config', 'r') as content_file:
    content = content_file.read()

print("total samples: ", len(names_data))

min_samples = 36
max_samples = 36
# build the model with model_builder
pipeline_proto = pipeline_pb2.TrainEvalPipelineConfig()
text_format.Merge(content, pipeline_proto)
model = model_builder.build(pipeline_proto.model, is_training=False)

# construct a network using the model
image_placeholder = tf.placeholder(shape=(None,None,3), dtype=tf.uint8, name='input')
original_image = tf.expand_dims(image_placeholder, 0)
preprocessed_image, true_image_shapes = model.preprocess(tf.to_float(original_image))
prediction_dict = model.predict(preprocessed_image, true_image_shapes)
detections = model.postprocess(prediction_dict, true_image_shapes)

# create an input network to read a file
filename_placeholder = tf.placeholder(name='file_name', dtype=tf.string)
image_file = tf.read_file(filename_placeholder)
image_data = tf.cond(tf.image.is_jpeg(image_file),
             lambda: tf.image.decode_jpeg(image_file, channels=3),
             lambda: tf.image.decode_png(image_file, channels=3))
#image_data = tf.image.decode_image(image_decoded)

# load the variables from a checkpoint
init_saver = tf.train.Saver()
sess = tf.Session()
# init_saver.restore(sess, 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/model.ckpt')
init_saver.restore(sess, 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/model.ckpt')


for ind,d in enumerate(names_data):

    print("image_num: ", ind)
    try: 
        d = d.rstrip().split("\t")[0]
        row_name = d.rstrip().split("\t")[0]
        d = img_root_folder + d
        image_name = d

        if not os.path.isfile(d):
            print("not exist", d)
            continue
        image_name = d
        #image_name = "161467.jpg"
        print("image name: ", d, " index: ", ind)
        # image_filename = 'COCO_val2014_000000224477.jpg'
        blob = sess.run(image_data, feed_dict={filename_placeholder: image_name})
        # process the inference
        output = sess.run(detections, feed_dict={image_placeholder:blob})
        # print(output)
    
        detection_scores = output["detection_scores"][0]
        detection_boxes = output['detection_boxes'][0]
        detection_features = output['detection_features'][0]
        detection_classes = output["detection_classes"][0]
    except Exception as e:
        with open("problem_images_v1.txt", "a") as f:
            f.write(image_name + "\n")
        print(e)
        continue
        #break
    detection_indices = np.argwhere(detection_scores > 0.5)
    print("detection indices length: " ,len(detection_indices))

    new_feature = {}
    if len(detection_indices) < min_samples:
        new_feature["detection_scores"] = detection_scores[0:min_samples]
        new_feature["detection_boxes"] = detection_boxes[0:min_samples]
        new_feature["detection_features"] = detection_features[0:min_samples]
        new_feature["detection_classes"] = detection_classes[0:min_samples]
    elif len(detection_indices) > max_samples:
        detection_indices = detection_indices[0:max_samples]
        new_feature["detection_scores"] = detection_scores[detection_indices]
        new_feature["detection_boxes"] = detection_boxes[detection_indices]
        new_feature["detection_features"] = detection_features[detection_indices]
        new_feature["detection_classes"] = detection_classes[detection_indices]
    else:
        new_feature["detection_scores"] = detection_scores[detection_indices]
        new_feature["detection_boxes"] = detection_boxes[detection_indices]
        new_feature["detection_features"] = detection_features[detection_indices]
        new_feature["detection_classes"] = detection_classes[detection_indices]

    print("after detection score length: ", len(new_feature["detection_scores"]))
    pickle.dump(new_feature, open("/data/shared/ConceptualCaptions/fastrcnn_features_all/" + image_name.split("/")[-1].split(".")[0].split("_")[0] + ".pkl", "wb"))
    print(image_name.split(".")[0].split("_")[0])
