import "../../index.css";
import { useRef, useState } from "@lynx-js/react";
import ImageCard from "./ImageCard.tsx";
import type { Picture } from "../../Pictures/faces/facesPictures.tsx";


import UploadIcon from "./UploadIcon.tsx";
import ReloadIcon from "./ReloadIcon.tsx";
const PLACEHOLDER_LABEL = 80;

type ModalProps = {
  onClose: () => void;
  picture: Picture;
}

// Store the final rendered dimension of the expanded image
let renderedTop: number;
let renderedBottom: number;
let renderedLeft: number;
let renderedRight: number;

function isBoxBigEnough(firstXY: number[], secondXY: number[]) {
  const deltaX = Math.abs(firstXY[0] - secondXY[0]);
  const deltaY = Math.abs(firstXY[1] - secondXY[1]);
  const threshold = 50; //  magic value
  return (deltaX > threshold && deltaY > threshold);
}

export default function Modal({ onClose, picture }: ModalProps) {
  const introText = "1) Tap screen once to see detected censored faces!\n\n2) Touch and drag to draw a new\
 bounding box to censor.\n\n3) Tap refresh to remove bounding boxes\n\n4) Tap outside the image to close.";
  const reRender = useState(false)[1]; // We don't use the state variable, just the reRender function
  let firstXY = useRef<[number, number]>([0, 0]);
  let secondXY = useRef<[number, number]>([0, 0]);

  return (
    <view
      className="modal"
      bindtap={onClose} // click backdrop closes modal
      style="top: 0px; left: 0px; width: 100%; aspectRatio: picture.width / picture.height; background-color:rgb(213, 128, 18);"
    >
      <view className="modal-content"
        bindlayoutchange={(e) => {
          renderedTop = e.detail.top;
          renderedBottom = e.detail.bottom;
          renderedLeft = e.detail.left;
          renderedRight = e.detail.right;
        }}
        catchtouchstart={(e) => {
          const x = e.touches[0]?.pageX || 0;
          const y = e.touches[0]?.pageY || 0;
          firstXY.current = [x, y];
        }

        }
        catchtouchend={(e) => {
          const x = e.touches[0]?.pageX || 0;
          const y = e.touches[0]?.pageY || 0;
          secondXY.current = [x, y];
          reRender((prev) => !prev)

          if (isBoxBigEnough(firstXY.current, secondXY.current)) {
            if (!picture.added_bounding_boxes || !picture.added_labels) {
              picture.added_bounding_boxes = [];
              picture.added_labels = [];
            }
            const normalX1 = (firstXY.current[0] - renderedLeft) / renderedRight; // account for 20 padding
            const normalY1 = (firstXY.current[1] - renderedTop) / renderedBottom;
            const normalX2 = (secondXY.current[0]) / renderedRight;
            const normalY2 = (secondXY.current[1]) / renderedBottom;

            // We need to ensure x1,y1 is the top left and x2,y2 the bottom right
            const smallerX1 = Math.max(0, Math.min(normalX1, normalX2));
            const smallerY1 = Math.max(0, Math.min(normalY1, normalY2));
            const largerX2 = Math.min(1, Math.max(normalX1, normalX2));
            const largerY2 = Math.min(1, Math.max(normalY1, normalY2));

            picture.added_bounding_boxes.push([smallerX1, smallerY1, largerX2, largerY2]);
            picture.added_labels.push(PLACEHOLDER_LABEL);
          }
        }

        }
        catchtap={(e) => null} // Catch tap so modal does not close when bubbling
      >
        <ImageCard picture={picture} />
        <UploadIcon picture={picture} />
        <ReloadIcon picture={picture} />
      </view>
      <text style="color: black; text-align: center; margin-top: 10px;">
        {introText}
      </text>
    </view>
  );

}
