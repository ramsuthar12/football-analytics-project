def get_bound_box_centre(bound_box):
    x1, y1, x2, y2 = bound_box
    return int((x1+x2)/2), int((y1+y2)/2)

def bound_box_width(bound_box):
    return bound_box[2] - bound_box[0]