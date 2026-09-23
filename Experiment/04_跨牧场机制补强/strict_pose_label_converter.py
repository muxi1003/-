"""Opt-in converter: reject contradictory labels instead of hiding visible points."""
import math


class AnnotationConflict(ValueError):
    pass


def convert_pose(data):
    width, height = data.get('imageWidth'), data.get('imageHeight')
    if not isinstance(width, (int,float)) or not isinstance(height, (int,float)) or min(width,height) <= 0:
        raise AnnotationConflict('invalid image dimensions')
    boxes = []; points = []
    for shape in data.get('shapes', []):
        label, kind, raw = shape.get('label'), shape.get('shape_type'), shape.get('points', [])
        if label not in ['nose','left_nostril','right_nostril']:
            raise AnnotationConflict(f'unknown label: {label}')
        if not raw or any(len(p)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) for p in raw):
            raise AnnotationConflict('invalid point coordinates')
        if any(x<0 or y<0 or x>width or y>height for x,y in raw):
            raise AnnotationConflict('coordinates outside image')
        if kind=='rectangle' and label=='nose' and len(raw) in [2,4]:
            x0=min(p[0] for p in raw); x1=max(p[0] for p in raw)
            y0=min(p[1] for p in raw); y1=max(p[1] for p in raw)
            if x1<=x0 or y1<=y0: raise AnnotationConflict('degenerate nose box')
            boxes.append(dict(group=shape.get('group_id'),box=(x0,y0,x1,y1),points={}))
        elif kind=='point' and label!='nose' and len(raw)==1:
            points.append(dict(group=shape.get('group_id'),label=label,point=raw[0]))
        else:
            raise AnnotationConflict('unsupported label/shape combination')
    if points and not boxes:
        raise AnnotationConflict('nostril point without nose box')
    for p in points:
        eligible=[b for b in boxes if b['group']==p['group']]
        if not eligible:
            # Missing group metadata is recoverable only with one containing box.
            x,y=p['point']
            eligible=[b for b in boxes
                      if (p['group'] is None or b['group'] is None)
                      and b['box'][0]<=x<=b['box'][2] and b['box'][1]<=y<=b['box'][3]]
        if len(eligible)!=1:
            raise AnnotationConflict('ambiguous or missing instance group')
        b=eligible[0]; x,y=p['point']; x0,y0,x1,y1=b['box']
        if p['label'] in b['points']:
            raise AnnotationConflict('duplicate nostril label in instance')
        if not (x0<=x<=x1 and y0<=y<=y1):
            raise AnnotationConflict('annotated nostril outside associated nose box')
        b['points'][p['label']]=(x,y)
    rows=[]
    for b in boxes:
        x0,y0,x1,y1=b['box']
        row=[0,(x0+x1)/(2*width),(y0+y1)/(2*height),(x1-x0)/width,(y1-y0)/height]
        for label in ['left_nostril','right_nostril']:
            if label in b['points']:
                x,y=b['points'][label]; row.extend([x/width,y/height,2])
            else:
                row.extend([0,0,0])
        rows.append(row)
    return rows


def serialize_pose(rows):
    return ''.join(' '.join(f'{v:.8f}' if i not in [0,7,10] else str(int(v))
                           for i,v in enumerate(row))+'\n' for row in rows)
