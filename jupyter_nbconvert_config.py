from traitlets.config import get_config
c = get_config()

c.SlidesExporter.reveal_config = {
    'center': False,         # Kill centering
    'transition': 'none',    # Faster loading
    'width': "100%",         # Use full browser width
    'height': "100%",        # Use full browser height
    'margin': 0,             # No margin
    'minScale': 1.0,         # Don't shrink content
    'maxScale': 1.0          # Don't grow content
}