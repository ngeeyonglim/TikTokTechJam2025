import "../index.css";
import type { Picture } from "../Pictures/faces/facesPictures.tsx";
import { useState } from "@lynx-js/react";
import  ModalProps  from "../Components/Modal.tsx";

import ImageCard from "../Components/ImageCard.tsx";
import { calculateEstimatedSize } from "./utils.tsx";

const SERVER_URL = "http://192.168.0.245:5000";
const POST_CENSORED_IMAGE_ENDPOINT = "/detect";
const LABEL_TO_CENSOR = 80 // 80 is for "face" in Yolov8 dataset

interface Detection {
  class_id: number;
  bbox: number[];
}

async function callLocalModelForCensor(label: number, image_path: string): Promise<Detection[] | undefined> {
  const URI = SERVER_URL + POST_CENSORED_IMAGE_ENDPOINT;
  try {
    const res = await fetch(URI, {
      method: "POST",
      headers: { "Content-Type": "application/json"},
      body: JSON.stringify({
        label: label,
        image: image_path
      })
    });
    const detections: Detection[] = await res.json(); // result will be an array of arrays
    return detections;
  } catch (err) {
    console.error("Error calling Python model:", err);
  }
}

export const Gallery = (props: { pictureData: Picture[] }) => {
  const { pictureData } = props;
  const [expanded, setExpanded] = useState<number | null>(null);
  const [isModalOpen, setModalOpen] = useState(false);
  if (isModalOpen) {
    const picture=pictureData[expanded??0];
    
    // Usage
    const detections = callLocalModelForCensor(LABEL_TO_CENSOR, picture.localSrc);
    detections.then((dets) => {
      picture.detected_bounding_boxes = [];
      dets?.forEach((det) => {
        picture.detected_bounding_boxes?.push(det.bbox);
      });
  });

    return(
      <view>
          <ModalProps picture={picture} onClose={() => setModalOpen(false)}>
          </ModalProps>
      </view>
    );
  } else {

  return (
    <view className="gallery-wrapper">
      <list
        className="list"
        list-type="waterfall"
        column-count={2}
        scroll-orientation="vertical"
        custom-list-name="list-container"
      >
        { pictureData.map((picture: Picture, index: number) => (
          <list-item
            estimated-main-axis-size-px={calculateEstimatedSize(picture.width, picture.height)}
            item-key={"" + index}
            key={"" + index}
            bindtap={() => {setExpanded(index); setModalOpen(true);}}
          >
            <ImageCard picture={picture} />
          </list-item>
        )) }
      </list>
      
    </view>
  );
  }
};

export default Gallery;