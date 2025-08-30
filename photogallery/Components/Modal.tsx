import "../index.css";
import { useRef, useState } from "@lynx-js/react";
import ImageCard from "./ImageCard.tsx";
import type { Picture } from "../Pictures/faces/facesPictures.tsx";


import UploadIcon from "./UploadIcon.tsx";
import ReloadIcon from "./ReloadIcon.tsx";
const PLACEHOLDER_LABEL = 80;

type ModalProps = {
  onClose: () => void;
  picture: Picture; // L
}

function isBoxDrawn(firstXY: number[], secondXY: number[]) {
  const deltaX = Math.abs(firstXY[0] - secondXY[0]);
  const deltaY = Math.abs(firstXY[1] - secondXY[1]);
  const threshold = 50; //  magic value
  return (deltaX > threshold && deltaY > threshold);
}

export default function Modal({ onClose, picture }: ModalProps) {
  const [xy, setXY] = useState<string | null>("1) Tap screen once to see detected censored faces!\n2) Touch and drag to draw a new\
 bounding box to censor.\n3) Tap refresh to remove bounding boxes\n4) Tap outside the image to close.");
  let firstXY = useRef<[number, number]>([0, 0]);
  let secondXY = useRef<[number, number]>([0, 0]);
  return (
    <view
      className="modal"
      bindtap={onClose} // click backdrop closes modal
      style="top: 0px; left: 0px; width: 400px; aspectRatio: picture.width / picture.height; background-color:rgb(53, 5, 74);"
    >
         <view className="modal-content"
         catchtouchstart={(e) => {
            const x = e.touches[0]?.pageX || 0;
            const y = e.touches[0]?.pageY || 0;
            const numTouches = e.touches.length;
            firstXY.current = [x, y];
  }
   
}
         catchtouchend={(e) => {
          const x = e.touches[0]?.pageX || 0;
          const y = e.touches[0]?.pageY || 0;
          const numTouches = e.touches.length;
          secondXY.current = [x, y];
          if (isBoxDrawn(firstXY.current, secondXY.current)){
            setXY("Bounding box drawn!")
          if (!picture.added_bounding_boxes || !picture.added_labels) {
            picture.added_bounding_boxes = [];
            picture.added_labels = [];
          }
          const normalX1 = (firstXY.current[0] - 20) / 360; // 400 width, 20 padding
          const normalY1 = (firstXY.current[1] - 20) / (360 / picture.width * picture.height); // height based on aspect ratio
          const normalX2 = (secondXY.current[0] - 20) / 360;
          const normalY2 = (secondXY.current[1] - 20) / (360 / picture.width * picture.height);

          // We need to ensure x1,y1 is the top left and x2,y2 the bottom right
          const smallerX1 = Math.min(normalX1, normalX2);
          const smallerY1 = Math.min(normalY1, normalY2);
          const largerX2 = Math.max(normalX1, normalX2);
          const largerY2 = Math.max(normalY1, normalY2);

          picture.added_bounding_boxes.push([smallerX1, smallerY1, largerX2, largerY2]);
          picture.added_labels.push(PLACEHOLDER_LABEL);
          }
        }

         }
           catchtap={(e) => setXY(xy)} // Catch tap so modal does not close when bubbling
           >
          <ImageCard picture={picture} />
          <UploadIcon picture={picture}/>
          <ReloadIcon picture={picture}/>
         </view>
         <text style="color: black; text-align: center; margin-top: 10px;">
            {xy}
          </text>
       </view>
  );
  
}
