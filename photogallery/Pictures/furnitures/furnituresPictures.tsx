import pic0 from "./0.png";
import pic1 from "./1.png";
import pic10 from "./10.png";
import pic11 from "./11.png";
import pic12 from "./12.png";
import pic13 from "./13.png";
import pic14 from "./14.png";
import pic2 from "./2.png";
import pic3 from "./3.png";
import pic4 from "./4.png";
import pic5 from "./5.png";
import pic6 from "./6.png";
import pic7 from "./7.png";
import pic8 from "./8.png";
import pic9 from "./9.png";

export interface Picture {
  src: string;
  localSrc: string;
  width: number;
  height: number;
  detected_bounding_boxes?: number[][] ; // x1, y1, x2, y2 (normalised to 0-1)
  added_bounding_boxes?: number[][] ; // We remeber the boxes that were user added, so
  // that we don't send back the model boxes (prevent over tuning)
  added_labels?: number[]; // Labels for the user added boxes
}
const BASE_PATH = "/home/don/ttjamLynx/photogallery/Pictures/furnitures/";
export const furnituresPicturesSubArray: Picture[] = [
  {
    src: pic0,
    localSrc: BASE_PATH + "0.png",
    width: 512,
    height: 429,
  },
  {
    src: pic1,
    localSrc: BASE_PATH+"1.png",
    width: 511,
    height: 437,
  },
  {
    src: pic2,
    localSrc: BASE_PATH+"2.png",
    width: 1024,
    height: 1589,
  },
  {
    src: pic3,
    localSrc: BASE_PATH+"3.png",
    width: 510,
    height: 418,
  },
  {
    src: pic4,
    localSrc: BASE_PATH+"4.png",
    width: 509,
    height: 438,
  },
  {
    src: pic5,
    localSrc: BASE_PATH+"5.png",
    width: 1024,
    height: 1557,
  },
  {
    src: pic6,
    localSrc: BASE_PATH+"6.png",
    width: 509,
    height: 415,
  },
  {
    src: pic7,
    localSrc: BASE_PATH+"7.png",
    width: 509,
    height: 426,
  },
  {
    src: pic8,
    localSrc: BASE_PATH+"8.png",
    width: 1024,
    height: 1544,
  },
  {
    src: pic9,
    localSrc: BASE_PATH+"9.png",
    width: 510,
    height: 432,
  },
  {
    src: pic10,
    localSrc: BASE_PATH+"10.png",
    width: 1024,
    height: 1467,
  },
  {
    src: pic11,
    localSrc: BASE_PATH+"11.png",
    width: 1024,
    height: 1545,
  },
  {
    src: pic12,
    localSrc: BASE_PATH+"12.png",
    width: 512,
    height: 416,
  },
  {
    src: pic13,
    localSrc: BASE_PATH+"13.png",
    width: 1024,
    height: 1509,
  },
  {
    src: pic14,
    localSrc: BASE_PATH+"14.png",
    width: 512,
    height: 411,
  },
];

export const furnituresPictures: Picture[] = [
  ...furnituresPicturesSubArray,
  ...furnituresPicturesSubArray,
  ...furnituresPicturesSubArray,
  ...furnituresPicturesSubArray,
  ...furnituresPicturesSubArray,
];
