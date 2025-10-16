EXTENSION_LIST = ["PNG", "png", "jpeg", "JPEG", "jpg", "JPG"]

HTML_TEMPLATE = """

<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
        }}
        #plotDiv {{
            width: 100%;
            height: 600px;
            margin-bottom: 20px;
        }}
        #imageGrid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .image-item {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .image-item:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .image-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-item .filename {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
            word-wrap: break-word;
            text-align: center;
        }}
        .placeholder {{
            grid-column: 1 / -1;
            text-align: center;
            color: #999;
            padding: 40px;
            font-size: 16px;
        }}
        #selectionInfo {{
            padding: 10px 20px;
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div id="plotDiv"></div>
    <div id="selectionInfo" style="display:none;">
        <strong>Selected points: <span id="selectionCount">0</span></strong> 
        (select area on the plot to view images)
    </div>
    <div id="imageGrid">
        <div class="placeholder">Select area on plot, first</div>
    </div>
    
    <script>
        var data = {data_json};
        
        var trace = {{
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: 8,
                color: data.x.map((v, i) => i),
                colorscale: 'Viridis',
                showscale: true,
                line: {{
                    width: 0.5,
                    color: 'white'
                }}
            }},
            text: data.filename,
            hovertemplate: '<b>%{{text}}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
        }};
        
        var layout = {{
            title: '2D t-SNE Visualization of Image Embeddings (Select to view images)',
            xaxis: {{
                title: 't-SNE Dimension 1',
                showgrid: true,
                gridcolor: 'lightgray'
            }},
            yaxis: {{
                title: 't-SNE Dimension 2',
                showgrid: true,
                gridcolor: 'lightgray'
            }},
            hovermode: 'closest',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            dragmode: 'select'
        }};
        
        var config = {{
            displayModeBar: true,
            modeBarButtonsToAdd: ['select2d', 'lasso2d']
        }};
        
        Plotly.newPlot('plotDiv', [trace], layout, config);
        
        var plotDiv = document.getElementById('plotDiv');
        var imageGrid = document.getElementById('imageGrid');
        var selectionInfo = document.getElementById('selectionInfo');
        var selectionCount = document.getElementById('selectionCount');
        
        function showImages(pointIndices) {{
            if (pointIndices.length === 0) {{
                imageGrid.innerHTML = '<div class="placeholder">Выделите область на графике, чтобы увидеть изображения</div>';
                selectionInfo.style.display = 'none';
                return;
            }}
            
            selectionInfo.style.display = 'block';
            selectionCount.textContent = pointIndices.length;
            
            imageGrid.innerHTML = '';
            pointIndices.forEach(idx => {{
                var item = document.createElement('div');
                item.className = 'image-item';
                item.innerHTML = `
                    <img src="${{data.img_b64[idx]}}" />
                    <div class="filename">${{data.filename[idx]}}</div>
                `;
                imageGrid.appendChild(item);
            }});
        }}
        
        plotDiv.on('plotly_selected', function(eventData) {{
            if (eventData && eventData.points) {{
                var indices = eventData.points.map(pt => pt.pointIndex);
                showImages(indices);
            }}
        }});
        
        plotDiv.on('plotly_deselect', function() {{
            showImages([]);
        }});
    </script>
</body>
</html>
"""
