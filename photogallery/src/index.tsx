import { facesPictures } from "../Pictures/faces/facesPictures.tsx";

import { root } from "@lynx-js/react";
import Gallery from "./Gallery.tsx";

function PictureList() {
  return <view className="gallery-wrapper multi-card">
  <Gallery pictureData={facesPictures} />;

  </view>
}

root.render(<PictureList />);

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
}
