# Failure-gallery image attribution and redistribution notice

All twelve PNG files in this directory are modified diagnostic overlays derived
from the same underlying photograph and may not be treated as unrestricted or
MIT-licensed project artwork.

## Underlying photograph

- **Title:** “Kitchen”
- **Creator:** Pot Noodle (`maggiew` on Flickr)
- **Original photograph:** https://www.flickr.com/photos/maggiew/6255196340/
- **Dataset reference:** COCO val2017 image `397133`
- **License:** Creative Commons Attribution 2.0
- **License text:** https://creativecommons.org/licenses/by/2.0/
- **Changes made:** resized/cropped only as performed by the preserved COCO
  asset, converted to PNG, and overlaid with colored reference/prediction masks,
  labels, and diagnostic text.

Redistribution of these PNGs requires preserving the creator credit, original
photo link, license link, and modification notice above. The repository's MIT
license does not apply to the underlying photograph.

## COCO annotations and model-derived overlays

Reference annotation geometry comes from the COCO Consortium annotations,
licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.
Model-derived masks were produced with the preserved Mask2Former checkpoint.
That checkpoint is documented by this project as CC BY-NC 4.0; this repository
does not grant broader commercial rights in the checkpoint or claim that the
diagnostic overlays remove upstream obligations. The gallery is retained as
noncommercial research/audit evidence.
