import { useRef, useState } from "@lynx-js/react";
import "../index.css";
import ImageCard from "./ImageCard.tsx";
import type { Picture } from "../Pictures/furnitures/furnituresPictures.tsx";

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
  const [xy, setXY] = useState<string | null>("hello!");
  let firstXY = useRef<[number, number]>([0, 0]);
  let secondXY = useRef<[number, number]>([0, 0]);
  return (
    <view
      className="modal"
      bindtap={onClose} // click backdrop closes modal
      style="top: 0px; left: 0px; width: 400px; aspectRatio: picture.width / picture.height; background-color:rgb(191, 90, 90);"
    >
         <view className="modal-content"
         catchtouchstart={(e) => {
            const x = e.touches[0]?.pageX || 0;
            const y = e.touches[0]?.pageY || 0;
            const numTouches = e.touches.length;
            setXY(`Num touch: ${numTouches}, Coordinates:  ${x.toFixed(1)}, ${y.toFixed(1)}`);
            firstXY.current = [x, y];
  }
   
}
         catchtouchend={(e) => {
          const x = e.touches[0]?.pageX || 0;
          const y = e.touches[0]?.pageY || 0;
          const numTouches = e.touches.length;
          secondXY.current = [x, y];
          if (isBoxDrawn(firstXY.current, secondXY.current)){
            setXY(`
              ${firstXY.current[0]}, ${firstXY.current[1]}
              ${secondXY.current[0]}, ${secondXY.current[1]}
            `)
          if (!picture.added_bounding_boxes || !picture.added_labels) {
            picture.added_bounding_boxes = [];
            picture.added_labels = [];
          }
          const normalX1 = (firstXY.current[0] - 20) / 360; // 400 width, 20 padding
          const normalY1 = (firstXY.current[1] - 20) / (360 / picture.width * picture.height); // height based on aspect ratio
          const normalX2 = (secondXY.current[0] - 20) / 360;
          const normalY2 = (secondXY.current[1] - 20) / (360 / picture.width * picture.height);
          picture.added_bounding_boxes.push([normalX1, normalY1, normalX2, normalY2]);
          picture.added_labels.push(PLACEHOLDER_LABEL);
          }
            // postBoundingBoxes(bounding_boxes, picture.src);
        }

         }
           catchtap={(e) => null} // Catch tap so modal does not close when bubbling
           >
          <ImageCard picture={picture} />
         </view>
         <text style="color: black; text-align: center; margin-top: 10px;">
            {xy}
          </text>
       </view>
  );
  
}
