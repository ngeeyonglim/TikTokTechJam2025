import pic12856 from "./wider_12856.jpg";
import pic12857 from "./wider_12857.jpg";
import pic12858 from "./wider_12858.jpg";
import pic12859 from "./wider_12859.jpg";
import pic12860 from "./wider_12860.jpg";
import pic12861 from "./wider_12861.jpg";
import pic12862 from "./wider_12862.jpg";
import pic12863 from "./wider_12863.jpg";
import pic12864 from "./wider_12864.jpg";
import pic12865 from "./wider_12865.jpg";
import pic12866 from "./wider_12866.jpg";
import pic12867 from "./wider_12867.jpg";
import pic12868 from "./wider_12868.jpg";
import pic12869 from "./wider_12869.jpg";
import pic12870 from "./wider_12870.jpg";
import pic12871 from "./wider_12871.jpg";
import pic12872 from "./wider_12872.jpg";
import pic12873 from "./wider_12873.jpg";
import pic12874 from "./wider_12874.jpg";
import pic12875 from "./wider_12875.jpg";
import pic12876 from "./wider_12876.jpg";
import pic12877 from "./wider_12877.jpg";
import pic12878 from "./wider_12878.jpg";
import pic12879 from "./wider_12879.jpg";

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
const BASE_PATH = "../Pictures/faces/";
export const facesPicturesSubArray: Picture[] = [
    {
        src: pic12856,
        localSrc: BASE_PATH + "wider_12856.jpg",
        width: 1024,
        height: 790,
    },
    
    {
        src: pic12857,
        localSrc: BASE_PATH + "wider_12857.jpg",
        width: 1024,
        height: 1543,
    },
    
    {
        src: pic12858,
        localSrc: BASE_PATH + "wider_12858.jpg",
        width: 1024,
        height: 1536,
    },
    
    // {
    //     src: pic12859,
    //     localSrc: BASE_PATH + "wider_12859.jpg",
    //     width: 1024,
    //     height: 1024,
    // },
    
    // {
    //     src: pic12860,
    //     localSrc: BASE_PATH + "wider_12860.jpg",
    //     width: 1024,
    //     height: 1449,
    // },
    
    // {
    //     src: pic12861,
    //     localSrc: BASE_PATH + "wider_12861.jpg",
    //     width: 1024,
    //     height: 603,
    // },
    
    // {
    //     src: pic12862,
    //     localSrc: BASE_PATH + "wider_12862.jpg",
    //     width: 1024,
    //     height: 685,
    // },
    
    // {
    //     src: pic12863,
    //     localSrc: BASE_PATH + "wider_12863.jpg",
    //     width: 1024,
    //     height: 820,
    // },
    
    // {
    //     src: pic12864,
    //     localSrc: BASE_PATH + "wider_12864.jpg",
    //     width: 1024,
    //     height: 1536,
    // },
    
    // {
    //     src: pic12865,
    //     localSrc: BASE_PATH + "wider_12865.jpg",
    //     width: 1024,
    //     height: 740,
    // },
    
    // {
    //     src: pic12866,
    //     localSrc: BASE_PATH + "wider_12866.jpg",
    //     width: 1024,
    //     height: 673,
    // },
    
    // {
    //     src: pic12867,
    //     localSrc: BASE_PATH + "wider_12867.jpg",
    //     width: 1024,
    //     height: 768,
    // },
    
    // {
    //     src: pic12868,
    //     localSrc: BASE_PATH + "wider_12868.jpg",
    //     width: 1024,
    //     height: 1536,
    // },
    
    // {
    //     src: pic12869,
    //     localSrc: BASE_PATH + "wider_12869.jpg",
    //     width: 1024,
    //     height: 1473,
    // },
    
    // {
    //     src: pic12870,
    //     localSrc: BASE_PATH + "wider_12870.jpg",
    //     width: 1024,
    //     height: 1366,
    // },
    
    // {
    //     src: pic12871,
    //     localSrc: BASE_PATH + "wider_12871.jpg",
    //     width: 1024,
    //     height: 683,
    // },
    
    // {
    //     src: pic12872,
    //     localSrc: BASE_PATH + "wider_12872.jpg",
    //     width: 1024,
    //     height: 768,
    // },
    
    // {
    //     src: pic12873,
    //     localSrc: BASE_PATH + "wider_12873.jpg",
    //     width: 1024,
    //     height: 1438,
    // },
    
    // {
    //     src: pic12874,
    //     localSrc: BASE_PATH + "wider_12874.jpg",
    //     width: 1024,
    //     height: 1408,
    // },
    
    // {
    //     src: pic12875,
    //     localSrc: BASE_PATH + "wider_12875.jpg",
    //     width: 1024,
    //     height: 683,
    // },
    
    // {
    //     src: pic12876,
    //     localSrc: BASE_PATH + "wider_12876.jpg",
    //     width: 1024,
    //     height: 1353,
    // },
    
    // {
    //     src: pic12877,
    //     localSrc: BASE_PATH + "wider_12877.jpg",
    //     width: 1024,
    //     height: 1536,
    // },
    
    // {
    //     src: pic12878,
    //     localSrc: BASE_PATH + "wider_12878.jpg",
    //     width: 1024,
    //     height: 705,
    // },
    
    // {
    //     src: pic12879,
    //     localSrc: BASE_PATH + "wider_12879.jpg",
    //     width: 1024,
    //     height: 683,
    // },
];

export const facesPictures: Picture[] = [
  ...facesPicturesSubArray,
];
