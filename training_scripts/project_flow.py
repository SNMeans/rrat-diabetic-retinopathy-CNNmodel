import plotly.graph_objects as go
from PIL import Image
import base64
import os

# Workflow steps
steps = [
    "Authorized User Logs In",
    "Success Message / Redirect to Home",
    "Access Patient Dashboard",
    "Select Patient",
    "Upload Retina Scan",
    "Preprocess Image",
    "Run ResNet-50",
    "Display Prediction",
]

# List of image paths
original_image_paths = [
    r"C:\Users\sumin\code\ResNet\authlogin.png",
    r"C:\Users\sumin\code\ResNet\success.png",
    r"C:\Users\sumin\code\ResNet\dashboard.png",
    r"C:\Users\sumin\code\ResNet\selectpt.png",
    r"C:\Users\sumin\code\ResNet\upload.png",
    r"C:\Users\sumin\code\ResNet\preprocess.png",
    r"C:\Users\sumin\code\ResNet\runresnet.png",
    r"C:\Users\sumin\code\ResNet\display.png",
]

# Resize images and convert to Base64
base64_images = []
for path in original_image_paths:
    img = Image.open(path)
    img = img.resize((200, 200))  # Resize images to 200x200
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        base64_images.append(f"data:image/png;base64,{encoded}")

# Initialize Plotly figure
fig = go.Figure()

# Spacing parameters
x_spacing = 2  # Horizontal spacing
y_image = 1    # Y-coordinate for images
y_text = 0.6   # Y-coordinate for text below images

# Add images and text
for i, (step, image_base64) in enumerate(zip(steps, base64_images)):
    # Add image
    fig.add_layout_image(
        dict(
            source=image_base64,
            x=i * x_spacing,  # X-position
            y=y_image,        # Y-position for images
            xref="x",
            yref="y",
            sizex=1,  # Size of the image
            sizey=1,
            xanchor="center",
            yanchor="middle",
            layer="above",
        )
    )
    # Add text below the image
    fig.add_trace(
        go.Scatter(
            x=[i * x_spacing],
            y=[y_text],
            mode="text",
            text=[step],
            textposition="middle center",
            textfont=dict(size=16, color="darkblue", family="Arial Bold"),  # Larger, bold text
        )
    )

# Add arrows between steps
for i in range(len(steps) - 1):
    fig.add_trace(
        go.Scatter(
            x=[i * x_spacing + 0.5, (i + 1) * x_spacing - 0.5],  # Arrow from current to next step
            y=[y_image, y_image],  # Horizontal alignment at y=y_image
            mode="lines+markers",
            line=dict(color="gray", width=3, dash="solid"),
            marker=dict(symbol="arrow", size=15, color="gray"),  # Arrow markers
            hoverinfo="skip",
        )
    )

# Update layout
fig.update_layout(
    title=dict(
        text="EyeQ Workflow with Images",
        font=dict(size=26, color="darkblue"),  # Larger title
        x=0.5,  # Centered title
    ),
    showlegend=False,
    xaxis=dict(visible=False, range=[-1, len(steps) * x_spacing]),
    yaxis=dict(visible=False, range=[-0.5, 2]),
    plot_bgcolor="rgb(240,240,255)",  # Light blue background
)

# Show the figure
fig.show()
