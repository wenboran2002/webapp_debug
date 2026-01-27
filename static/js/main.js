$(document).ready(function() {
    let totalFrames = 0;
    let currentFrame = 0;
    let isPlaying = false;
    let fps = 30; // Default fps, will be updated from video metadata
    let playInterval = null;
    let playAnimationFrameId = null; // For requestAnimationFrame
    let sliderUpdateTimeout = null; // For throttling slider updates
    let dragDebounceTimer = null; // For debouncing frame updates during dragging
    let meshData = null;
    let selectedPoints = []; // Object points
    let activeObjectPointIndex = -1; // The currently selected object point for annotation
    let humanKeypoints = {}; // { jointName: {index, x, y, z} }
    // New structure for annotations:
    // annotations[objPointIndex] = { type: 'human', joint: 'name' } OR { type: '2d', trackId: '...' }
    // But to keep it simple and compatible with existing save logic:
    // humanKeypoints maps Joint -> ObjPoint.
    // We need a map ObjPoint -> Joint to easily check status.
    // And ObjPoint -> 2DTrack.
    let objPointToJoint = {}; // { objIdx: jointName } for current frame only
    let objPointToTrack = {}; // { objIdx: { frameIdx: [x, y] } }

    // For 3D human-joint mapping, we want edits at a given
    // frame to only affect that frame and later frames, not
    // earlier frames. We therefore maintain per-object keyframes
    // over time and derive objPointToJoint for the current frame
    // from those keyframes.
    // jointKeyframesByObj[objIdx] = [ { frame: int, joint: string|null }, ... ]
    let jointKeyframesByObj = {};

    // Similarly, for 3D point visibility on the object surface,
    // we keep per-object visibility keyframes so that deleting
    // a point at a given frame hides it from that frame onward
    // without affecting earlier frames.
    // visibilityKeyframesByObj[objIdx] = [ { frame: int, visible: bool }, ... ]
    let visibilityKeyframesByObj = {};
    
    let currentMode = 'view'; // 'view', 'select', 'delete'
    let selectedJointName = null;
    let jointTree = null;
    let mainJointCoords = null;
    let buttonNames = null;

    // 2D Annotation variables
    let pending2DPoint = null; // last clicked 2D point (for legacy use / highlighting)
    let pending2DPoints = {};  // { objIdx: { x, y, displayX, displayY, frame } }

        // Frame preloading cache
        let preloadCache = new Map();

        // 预加载相邻帧（优化版本，使用缓存）
        function preloadFrames(frameNum) {
            const preloadCount = 2; // 减少预加载数量以节省内存
            const framesToPreload = [];

            // 收集需要预加载的帧
            for (let i = 1; i <= preloadCount; i++) {
                if (frameNum + i < totalFrames && !preloadCache.has(frameNum + i)) {
                    framesToPreload.push(frameNum + i);
                }
                if (frameNum - i >= 0 && !preloadCache.has(frameNum - i)) {
                    framesToPreload.push(frameNum - i);
                }
            }

            // 限制缓存大小，避免内存泄漏
            if (preloadCache.size > 50) {
                // 清除距离当前帧最远的缓存
                const sortedKeys = Array.from(preloadCache.keys()).sort((a, b) =>
                    Math.abs(a - frameNum) - Math.abs(b - frameNum)
                );
                // 保留最近的30个，删除其余的
                sortedKeys.slice(30).forEach(key => {
                    preloadCache.delete(key);
                });
            }

            // 预加载新帧
            framesToPreload.forEach(frameNum => {
                const img = new Image();
                img.onload = () => {
                    // 图像加载完成后加入缓存
                    preloadCache.set(frameNum, img);
                };
                img.src = '/api/frame/' + frameNum;
            });
        }
    
    // Plotly variables
    let meshTrace = null;
    let scatterTrace = null; // Object points (Red)
    let humanTrace = null;   // Human keypoints (Green)
    let layout = null;

    // Initialize
    loadJointTree();
    loadHumanSelectorData();
    fetchMetadata();
    loadMesh();

    // Optimization Button Handler
    $('#btn-optimize').click(function() {
        const btn = $(this);
        btn.prop('disabled', true);
        
        // 1. Save merged annotations first
        saveMergedAnnotations(function(success) {
            if (!success) {
                alert('Failed to save annotations before optimization.');
                btn.prop('disabled', false);
                return;
            }
            
            // 2. Run optimization
            $.ajax({
                url: '/api/run_optimization',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    frame_idx: currentFrame
                }),
                success: function(response) {
                    if (response.status === 'success') {
                        console.log('Optimization completed successfully!');
                        // Reload scene data to show updated results
                        updateSceneData(currentFrame);
                    } else {
                        alert('Optimization failed: ' + (response.message || 'Unknown error'));
                    }
                },
                error: function(xhr, status, error) {
                    console.error('Optimization error:', error);
                    alert('Error running optimization: ' + error);
                },
                complete: function() {
                    btn.prop('disabled', false);
                }
            });
        });
    });
    

    // Load existing annotations if any (mock for now, or fetch from server)
    // In a real app, we would fetch saved annotations here.
    // For now, we start fresh or rely on what's in memory if page isn't reloaded.

    function loadJointTree() {
        $.getJSON('/asset/data/joint_tree.json', function(data) {
            jointTree = data;
            initJointDropdown();
        });
    }

    function loadHumanSelectorData() {
        $.when(
            $.getJSON('/asset/data/main_joint.json'),
            $.getJSON('/asset/data/button_name.json')
        ).done(function(coords, names) {
            mainJointCoords = coords[0];
            buttonNames = names[0];
            initHumanKeypointSelector();
        });
    }

    function initHumanKeypointSelector() {
        const container = $('#human-image-container');
        const img = $('#human-ref-img');
        
        // Wait for image to load to get dimensions
        if (img[0].complete) {
            renderButtons();
        } else {
            img.on('load', renderButtons);
        }

        function renderButtons() {
            const naturalWidth = img[0].naturalWidth;
            const naturalHeight = img[0].naturalHeight;
            
            // Use a fixed reference size for scaling logic to match original coordinates
            // The original app likely used 480x480 as the display size for the coordinates.
            const referenceSize = 480;
            
            // Get current container dimensions
            const containerWidth = container.width();
            const containerHeight = container.height();
            
            // Calculate scale factor based on how much the container has shrunk/grown relative to reference
            // Assuming the image fills the container and maintains aspect ratio (square)
            const scaleFactor = containerWidth / referenceSize;
            
            // Clear existing buttons
            container.find('.joint-btn').remove();

            for (const [jointName, coords] of Object.entries(mainJointCoords)) {
                let [realX, realY] = coords;
                
                // Apply offsets from app_new.py logic (these are in original image coordinates?)
                // Or are they in the 480x480 space?
                // The original code: scaleX = (realX / naturalWidth) * displayWidth;
                // If naturalWidth is the original image width, and displayWidth is 480.
                // Let's stick to the proportional logic but use current container size.
                
                realX = realX - 40;
                realY = realY + 70;
                
                if (jointName === "leftNeck") {
                    realX = realX + 25;
                }
                if (jointName === "rightNeck") {
                    realX = realX - 25;
                }

                // Calculate position as percentage of container
                // This makes it responsive to any size
                let percentX = (realX / naturalWidth) * 100;
                let percentY = (realY / naturalHeight) * 100;
                
                // Adjust Y for specific joints (offset in pixels originally, now needs to be relative or scaled)
                // Original: scaleY -= 20 (pixels in 480px space)
                const legJoints = ["leftLowerLeg", "leftFoot", "rightLowerLeg", "rightFoot"];
                let yOffsetPercent = 0;
                if (legJoints.includes(jointName)) {
                    // 20px out of 480px is approx 4.16%
                    yOffsetPercent = (20 / 480) * 100; 
                    // Wait, original code subtracted 20 from scaleY.
                    // scaleY = (realY / naturalHeight) * displayHeight - 10;
                    // So total offset was -10 - 20 = -30 for legs.
                    // And -10 for others.
                }
                
                // Base offset -10px (approx 2%)
                let baseOffsetPercent = (10 / 480) * 100;
                
                percentY -= baseOffsetPercent;
                percentY -= yOffsetPercent;

                const btnLabel = buttonNames[jointName] || jointName;
                
                const btn = $('<div class="joint-btn"></div>')
                    .text(btnLabel)
                    .css({
                        left: percentX + '%',
                        top: percentY + '%',
                        fontSize: Math.max(10, 10 * scaleFactor) + 'px', // Scale font but min 10px
                        padding: Math.max(2, 2 * scaleFactor) + 'px',
                        position: 'absolute',
                        transform: 'translate(-50%, -50%)' // Center button on point
                    })
                    .attr('title', jointName);
                
                btn.click(function(e) {
                    e.stopPropagation();
                    showContextMenu(jointName, e.pageX, e.pageY);
                });

                container.append(btn);
            }
        }
        
        // Re-render on resize
        $(window).resize(function() {
             if (container.is(':visible')) renderButtons();
        });
        
        // Also re-render when tab is switched
        $('#tab-human').click(function() {
             setTimeout(renderButtons, 50);
        });
    }

    function showContextMenu(mainJoint, x, y) {
        // Remove existing context menu
        $('.context-menu').remove();

        const menu = $('<div class="context-menu"></div>');
        
        // Add sub-joints
        const subJoints = jointTree[mainJoint] || [];
        
        // Add main joint itself as an option
        const mainItem = $('<div class="context-menu-item"></div>')
            .text(mainJoint + " (Main)")
            .click(function() {
                selectJoint(mainJoint);
                menu.remove();
            });
        menu.append(mainItem);

        if (subJoints.length > 0) {
            menu.append('<div class="context-menu-separator"></div>');
            subJoints.forEach(sub => {
                const item = $('<div class="context-menu-item"></div>')
                    .text(sub)
                    .click(function() {
                        selectJoint(sub);
                        menu.remove();
                    });
                menu.append(item);
            });
        }
        
        // Add Cancel
        menu.append('<div class="context-menu-separator"></div>');
        const cancelItem = $('<div class="context-menu-item"></div>')
            .text("Cancel")
            .click(function() {
                menu.remove();
            });
        menu.append(cancelItem);

        $('body').append(menu);
        
        // Position menu
        menu.css({
            left: x + 'px',
            top: y + 'px',
            display: 'block'
        });

        // Close on click outside
        $(document).one('click', function() {
            menu.remove();
        });
    }

    function initJointDropdown() {
        // Deprecated dropdown logic, but we can use it to populate a tree view if needed.
        // For now, the visual selector is primary.
    }

    function selectJoint(name) {
        selectedJointName = name;
        $('#selected-joint-display').text(name);
        // If we have an active object point, automatically link it?
        // User might want to confirm. But previous logic was auto-click if active.
        // Let's keep it manual for now as per "Link to Joint" button presence.
        // Actually, user said "click human joint OR click 2D point".
        // So maybe auto-link is better?
        // "Link to Joint" button is there. Let's use it.
    }

    // ===== 3D Joint Mapping over Time (per-frame semantics) =====

    function addJointKeyframe(objIdx, frame, jointName) {
        if (!jointKeyframesByObj[objIdx]) {
            jointKeyframesByObj[objIdx] = [];
        }
        jointKeyframesByObj[objIdx].push({ frame: frame, joint: jointName });
        // Keep keyframes sorted by frame
        jointKeyframesByObj[objIdx].sort((a, b) => a.frame - b.frame);
    }

    function getJointForObjectAtFrame(objIdx, frame) {
        const kfs = jointKeyframesByObj[objIdx];
        if (!kfs || kfs.length === 0) return null;
        let result = null;
        for (const kf of kfs) {
            if (kf.frame <= frame) {
                result = kf.joint;
            } else {
                break;
            }
        }
        return result;
    }

    function applyJointMappingForCurrentFrame() {
        // Recompute objPointToJoint for the current frame from keyframes
        objPointToJoint = {};
        selectedPoints.forEach(pt => {
            const j = getJointForObjectAtFrame(pt.index, currentFrame);
            if (j) {
                objPointToJoint[pt.index] = j;
            }
        });
    }

    function addVisibilityKeyframe(objIdx, frame, visible) {
        if (!visibilityKeyframesByObj[objIdx]) {
            visibilityKeyframesByObj[objIdx] = [];
        }
        visibilityKeyframesByObj[objIdx].push({ frame: frame, visible: !!visible });
        visibilityKeyframesByObj[objIdx].sort((a, b) => a.frame - b.frame);
    }

    function isObjectVisibleAtFrame(objIdx, frame) {
        const kfs = visibilityKeyframesByObj[objIdx];
        if (!kfs || kfs.length === 0) return true; // default visible
        let result = true;
        for (const kf of kfs) {
            if (kf.frame <= frame) {
                result = kf.visible;
            } else {
                break;
            }
        }
        return result;
    }

    // Tab Switching Logic
    $('#tab-human').click(function() {
        $('.panel-tab').removeClass('active').css({
            'background': 'transparent',
            'color': '#666'
        });
        $(this).addClass('active').css({
            'background': 'white',
            'color': '#667eea'
        });
        $('#panel-human-joints').show();
        $('#panel-2d-view').hide();
    });

    $('#tab-2d').click(function() {
        $('.panel-tab').removeClass('active').css({
            'background': 'transparent',
            'color': '#666'
        });
        $(this).addClass('active').css({
            'background': 'white',
            'color': '#667eea'
        });
        $('#panel-human-joints').hide();
        $('#panel-2d-view').css('display', 'flex');
        init2DCanvas();
    });

    // Event Listeners
    $('#play-pause').click(togglePlay);
    
    // Use 'input' event for immediate response during dragging
    // Don't pause playback, just update frame immediately
    let isDragging = false;
    let lastUpdateTime = 0;
    const minUpdateInterval = 16; // ~60fps max update rate
    
    $('#frame-slider').on('mousedown', function() {
        isDragging = true;
        // Don't pause playback - let it continue, but slider will override
    });
    
    $('#frame-slider').on('input', function() {
        const newFrame = parseInt($(this).val());
        if (newFrame !== currentFrame) {
            // Update progress display immediately for responsive UI
            updateProgressDisplay(newFrame);

            // Clear previous debounce timer
            if (dragDebounceTimer) {
                clearTimeout(dragDebounceTimer);
            }

            // Set new debounce timer (100ms delay)
            dragDebounceTimer = setTimeout(function() {
                loadFrame(newFrame);
                dragDebounceTimer = null;
            }, 100);
        }
    });
    
    $('#frame-slider').on('mouseup', function() {
        isDragging = false;
        // Cancel any pending debounce timer and update immediately
        if (dragDebounceTimer) {
            clearTimeout(dragDebounceTimer);
            dragDebounceTimer = null;
        }
        currentFrame = parseInt($(this).val());
        loadFrame(currentFrame);
    });
    
    // Also handle change event for final update (touch devices)
    $('#frame-slider').on('change', function() {
        currentFrame = parseInt($(this).val());
        loadFrame(currentFrame);
    });
    // Save buttons: per-frame inside annotation modal, and global final-save button on main page
    $('#save-annotation').click(saveAnnotation);
    $('#save-annotation-main').click(saveAllAnnotations);

    // Mode Switching
    $('.mode-btn').click(function() {
        // Ignore special buttons
        if (['btn-set-joint', 'btn-show-selector', 'btn-track-2d', 'btn-manage', 'btn-toggle-2d'].includes(this.id)) return;
        
        $('.mode-btn').removeClass('active');
        $(this).addClass('active');
        
        if (this.id === 'mode-view') currentMode = 'view';
        else if (this.id === 'mode-select') currentMode = 'select';
        else if (this.id === 'mode-delete') currentMode = 'delete';
        
        // Update Plotly dragmode
        const dragmode = currentMode === 'view' ? 'orbit' : 'orbit'; // Always orbit, but click behavior changes
        Plotly.relayout('3d-viewer', { 'scene.dragmode': dragmode });
        
        console.log("Switched to mode:", currentMode);
    });

    // Removed 2D View Toggle logic as it is now tab-based

    function init2DCanvas() {
        const img = document.getElementById('modal-video-frame');
        const canvas = document.getElementById('modal-video-overlay');
        
        // Wait for image load if needed
        if (img.complete) {
            resizeCanvas();
        } else {
            img.onload = resizeCanvas;
        }
        
        function resizeCanvas() {
            // Set canvas size to match the image element's rendered size
            // Note: If object-fit: contain is used, clientWidth/Height includes black bars.
            // We should ideally match the actual image area, but for simplicity we match the element
            // and handle coordinate mapping in the click handler.
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
        }
        
        // Handle resize
        $(window).resize(resizeCanvas);
    }

    function runTrackingForObject(objIdx, startFrame, x, y, onDone, onError) {
        $.ajax({
            url: '/api/track_2d',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                frame_idx: startFrame,
                x: x,
                y: y
            }),
            success: function(response) {
                const filteredTracks = {};
                for (const [f, pt] of Object.entries(response.tracks || {})) {
                    const fi = parseInt(f);
                    if (!Number.isNaN(fi) && fi >= startFrame) {
                        filteredTracks[fi] = pt;
                    }
                }

                const existing = objPointToTrack[objIdx] || {};
                const merged = { ...existing };
                for (const [fi, pt] of Object.entries(filteredTracks)) {
                    merged[fi] = pt;
                }
                objPointToTrack[objIdx] = merged;

                if (objPointToJoint[objIdx]) {
                    delete objPointToJoint[objIdx];
                }

                updateFrame();
                updateSelection();
                if (onDone) onDone();
            },
            error: function(xhr) {
                if (onError) onError(xhr);
            }
        });
    }

    // Helper: clear 2D track for an object point only from
    // a given frame onward, keeping earlier frames intact.
    function clearTrackFromFrame(objIdx, fromFrame) {
        const track = objPointToTrack[objIdx];
        if (!track) return;

        const pruned = {};
        for (const [fStr, pt] of Object.entries(track)) {
            const fi = parseInt(fStr);
            if (Number.isNaN(fi) || fi < fromFrame) {
                pruned[fStr] = pt;
            }
        }

        if (Object.keys(pruned).length > 0) {
            objPointToTrack[objIdx] = pruned;
        } else {
            delete objPointToTrack[objIdx];
        }
    }

    // 2D Tracking Logic - single point
    $('#btn-track-2d').click(function() {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first.");
            return;
        }

        const info = pending2DPoints[activeObjectPointIndex];
        if (!info) {
            alert("Please click on the video frame to set a 2D point for this object point first.");
            return;
        }

        const btn = $(this);
        const btnAll = $('#btn-track-2d-all');
        btn.prop('disabled', true).text('Tracking...');
        btnAll.prop('disabled', true);
        $('#2d-status').text("Tracking in progress...");

        const objIdx = activeObjectPointIndex;
        runTrackingForObject(
            objIdx,
            info.frame,
            info.x,
            info.y,
            function() {
                delete pending2DPoints[objIdx];
                pending2DPoint = null;
                if (Object.keys(pending2DPoints).length === 0) {
                    btnAll.prop('disabled', true).text('Track All');
                }
                btn.prop('disabled', false).text('Track 2D Point');
                $('#2d-status').text("Tracking complete!");
                update2DOverlay();
            },
            function(xhr) {
                btn.prop('disabled', false).text('Track 2D Point');
                btnAll.prop('disabled', Object.keys(pending2DPoints).length === 0).text('Track All');
                $('#2d-status').text("Error: " + (xhr.responseJSON?.error || "Tracking failed"));
                alert("Tracking failed: " + (xhr.responseJSON?.error || "Unknown error"));
            }
        );
    });

    // 2D Tracking Logic - track all pending points using a single CoTracker call per frame
    $('#btn-track-2d-all').click(function() {
        const entries = Object.entries(pending2DPoints);
        if (entries.length === 0) {
            alert("No pending 2D points to track.");
            return;
        }

        const btnAll = $(this);
        const btnSingle = $('#btn-track-2d');
        btnAll.prop('disabled', true).text('Tracking All...');
        btnSingle.prop('disabled', true);
        $('#2d-status').text("Tracking all pending 2D points...");

        // Group pending points by their start frame so that
        // each CoTracker run can handle multiple queries at once.
        const frameGroups = {};
        for (const [objIdxStr, info] of entries) {
            const f = info.frame;
            if (!frameGroups[f]) {
                frameGroups[f] = [];
            }
            frameGroups[f].push({
                objIdx: parseInt(objIdxStr),
                x: info.x,
                y: info.y
            });
        }

        const frames = Object.keys(frameGroups)
            .map(f => parseInt(f))
            .sort((a, b) => a - b);

        let groupIndex = 0;
        let hadError = false;

        function processNextGroup() {
            if (groupIndex >= frames.length) {
                // All groups processed
                pending2DPoints = {};
                pending2DPoint = null;
                btnAll.prop('disabled', true).text('Track All');
                btnSingle.prop('disabled', false).text('Track 2D Point');
                $('#2d-status').text(hadError ? 'Finished with some errors' : 'All tracking complete!');
                updateFrame();
                updateSelection();
                update2DOverlay();
                return;
            }

            const frame = frames[groupIndex++];
            const points = frameGroups[frame];
            if (!points || points.length === 0) {
                processNextGroup();
                return;
            }

            $.ajax({
                url: '/api/track_2d_multi',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    frame_idx: frame,
                    points: points.map(p => ({ obj_idx: p.objIdx, x: p.x, y: p.y }))
                }),
                success: function(response) {
                    const tracksByObj = response.tracks || {};
                    for (const [objIdxKey, trackDict] of Object.entries(tracksByObj)) {
                        const objIdx = parseInt(objIdxKey);
                        const existing = objPointToTrack[objIdx] || {};
                        const merged = { ...existing };

                        for (const [fStr, pt] of Object.entries(trackDict)) {
                            const fi = parseInt(fStr);
                            if (!Number.isNaN(fi) && fi >= frame) {
                                merged[fi] = pt;
                            }
                        }

                        objPointToTrack[objIdx] = merged;

                        if (objPointToJoint[objIdx]) {
                            delete objPointToJoint[objIdx];
                        }

                        if (pending2DPoints[objIdx] !== undefined) {
                            delete pending2DPoints[objIdx];
                        }
                    }

                    // Update view after each group
                    updateFrame();
                    updateSelection();
                    update2DOverlay();

                    processNextGroup();
                },
                error: function(xhr) {
                    console.error('Multi-point tracking error for frame', frame, xhr.responseJSON || xhr.statusText);
                    hadError = true;
                    processNextGroup();
                }
            });
        }

        // Start processing frame groups sequentially
        processNextGroup();
    });

    // Canvas Click Handler (2D Selection)
    $('#modal-video-overlay').click(function(e) {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first!");
            return;
        }

        const rect = this.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        const img = document.getElementById('modal-video-frame');
        
        // Calculate actual image dimensions and offsets due to object-fit: contain
        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;
        
        let renderWidth, renderHeight, offsetX, offsetY;
        
        if (clientRatio > naturalRatio) {
            // Image is pillarboxed (black bars on sides)
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            // Image is letterboxed (black bars on top/bottom)
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }
        
        // Check if click is within the actual image
        if (clickX < offsetX || clickX > offsetX + renderWidth ||
            clickY < offsetY || clickY > offsetY + renderHeight) {
            // Clicked on black bars
            return;
        }
        
        // Map to natural coordinates
        const x = (clickX - offsetX) * (img.naturalWidth / renderWidth);
        const y = (clickY - offsetY) * (img.naturalHeight / renderHeight);
        
        // Store this click as a pending 2D point for the active object index
        pending2DPoints[activeObjectPointIndex] = {
            x: x,
            y: y,
            displayX: clickX,
            displayY: clickY,
            frame: currentFrame
        };
        // Keep last clicked for legacy use
        pending2DPoint = pending2DPoints[activeObjectPointIndex];

        const pendingCount = Object.keys(pending2DPoints).length;
        $('#2d-status').text(
            `Selected 2D for Obj Point ${activeObjectPointIndex}: (${Math.round(x)}, ${Math.round(y)}). ` +
            `Pending points: ${pendingCount}`
        );
        $('#btn-track-2d').prop('disabled', false);
        $('#btn-track-2d-all').prop('disabled', false);

        // Redraw overlay to show all pending points
        update2DOverlay();
    });
    
    // Human Joint Mapping Logic
    $('#btn-set-joint').click(function() {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first!");
            return;
        }
        if (!selectedJointName) {
            alert("Please select a joint from the dropdown first!");
            return;
        }
        
        // Add a joint mapping keyframe for this object from the
        // current frame onward (earlier frames remain unchanged).
        addJointKeyframe(activeObjectPointIndex, currentFrame, selectedJointName);
        applyJointMappingForCurrentFrame();
        
        // Clear any 2D track for this point (exclusive choice)
        if (objPointToTrack[activeObjectPointIndex]) {
            delete objPointToTrack[activeObjectPointIndex];
        }
        
        $('#selected-joint-display').text(selectedJointName + " (Linked)");
        updateSelection(); // Update 3D view
    });

    // Modal Controls
    $('#btn-annotate').click(openModal);
    $('#close-modal').click(closeModal);
    $(window).click(function(event) {
        if (event.target.id === 'annotation-modal') {
            closeModal();
        }
    });

    function openModal() {
        // Pause video if playing
        if (isPlaying) {
            togglePlay();
        }
        
        $('#modal-frame-idx').text(currentFrame);
        $('#annotation-modal').show();
        
        // Initialize 2D canvas
        init2DCanvas();

        // Ensure the 2D view image and overlay are synced to current frame
        // 调用一次 updateFrame 来给 modal-video-frame 设置 src 并绘制 2D 叠加
        updateFrame();
        
        // Trigger Plotly resize/redraw
        if (meshData) {
            Plotly.relayout('3d-viewer', {
                'width': $('#3d-viewer').width(),
                'height': $('#3d-viewer').height()
            });
        }
    }

    function closeModal() {
        $('#annotation-modal').hide();
    }

    function fetchMetadata() {
        $.get('/api/metadata', function(data) {
            totalFrames = data.total_frames;
            fps = data.fps || 30; // Use original video fps
            $('#frame-slider').attr('max', totalFrames - 1);
            
            // Update Info Panel
            $('#info-video').text(`Video: ${data.video_name}`);
            $('#info-object').text(`Object: ${data.obj_name}`);
            $('#info-res').text(`Res: ${data.width}x${data.height}`);
            $('#info-fps').text(`FPS: ${Math.round(fps)}`);
            $('#info-frames').text(`Frames: ${totalFrames}`);
            
            // Preload first frame
            loadFrame(0);
            
            // Preload next few frames for smoother playback
            preloadFrames(0, Math.min(5, totalFrames - 1));
        });
    }
    
    function preloadFrames(startFrame, endFrame) {
        // Preload frames in background with caching for better performance
        const framesToPreload = [];

        // Collect frames that aren't already cached
        for (let i = startFrame; i <= endFrame; i++) {
            if (!preloadCache.has(i)) {
                framesToPreload.push(i);
            }
        }

        // Limit cache size to prevent memory leaks (keep last 50 frames)
        if (preloadCache.size > 50) {
            const sortedKeys = Array.from(preloadCache.keys()).sort((a, b) =>
                Math.abs(a - currentFrame) - Math.abs(b - currentFrame)
            );
            // Remove oldest frames beyond the cache limit
            sortedKeys.slice(30).forEach(key => {
                preloadCache.delete(key);
            });
        }

        // Preload new frames
        framesToPreload.forEach(frameNum => {
            const img = new Image();
            img.onload = () => {
                // Cache the loaded image
                preloadCache.set(frameNum, img);
            };
            img.src = '/api/frame/' + frameNum + '?t=' + Date.now();
        });
    }
    
    // New Controls
    $('#static-object').change(function() {
        const isStatic = $(this).is(':checked');
        console.log("Static Object mode:", isStatic);
        // TODO: Send this state to backend if needed for tracking logic
    });
    
    // Scale slider debounce timer
    let scaleSliderDebounceTimer = null;

    // Function to apply scale (extracted for reuse)
    function applyScale(scale, updateSlider = true) {
        if (!(scale > 0)) {
            $('#scale-status').text('Scale must be a positive number');
            return;
        }

        // Update slider if requested
        if (updateSlider) {
            const clampedScale = Math.max(0.1, Math.min(5, scale));
            $('#slider-scale-factor').val(clampedScale);
            $('#slider-scale-value').text(clampedScale.toFixed(2));
        }

        $('#btn-apply-scale').prop('disabled', true).text('Applying...');
        $('#scale-status').text('Applying scale, please wait...');

        $.ajax({
            url: '/api/set_scale',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ scale_factor: scale }),
            success: function(resp) {
                $('#scale-status').text('Scale applied: ' + resp.scale_factor + 'x');
                $('#btn-apply-scale').prop('disabled', false).text('Apply Scale');
                // Reload viewer with updated object mesh, based on the same frame
                loadScaleViewer(currentFrame);
            },
            error: function(xhr) {
                $('#scale-status').text('Error: ' + (xhr.responseJSON?.error || 'Failed to apply scale'));
                $('#btn-apply-scale').prop('disabled', false).text('Apply Scale');
            }
        });
    }

    $('#btn-check-scale').click(function() {
        $('#scale-frame-idx').text(currentFrame);
        $('#scale-modal').show();
        
        $('#scale-status').text('');
        // Initialize slider and input with default value 1.0
        $('#input-scale-factor').val('1.0');
        $('#slider-scale-factor').val('1.0');
        $('#slider-scale-value').text('1.00');
        loadScaleViewer(currentFrame);
    });
    
    $('#close-scale-modal').click(function() {
        $('#scale-modal').hide();
    });
    
    $(window).click(function(event) {
        if (event.target.id === 'scale-modal') {
            $('#scale-modal').hide();
        }
    });

    function loadScaleViewer(frameIdx) {
        const viewer = document.getElementById('scale-viewer');

        // Show loading
        Plotly.newPlot('scale-viewer', [], {
            title: 'Loading Scene Data...',
            xaxis: { visible: false },
            yaxis: { visible: false }
        });

        $.get('/api/scene_data/' + frameIdx, function(data) {
            const human = data.human;
            const object = data.object;

            const humanTrace = {
                type: 'mesh3d',
                x: human.x, y: human.y, z: human.z,
                i: human.i, j: human.j, k: human.k,
                color: 'pink', opacity: 1.0,
                name: 'Human'
            };

            const objectTrace = {
                type: 'mesh3d',
                x: object.x, y: object.y, z: object.z,
                i: object.i, j: object.j, k: object.k,
                color: 'lightblue', opacity: 0.8,
                name: 'Object'
            };

            const layout = {
                scene: {
                    aspectmode: 'data',
                    dragmode: 'orbit'
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                showlegend: true
            };

            Plotly.newPlot('scale-viewer', [humanTrace, objectTrace], layout, { responsive: true });

        }).fail(function(xhr) {
            alert('Error loading scene data: ' + (xhr.responseJSON?.error || 'Unknown error'));
            $('#scale-modal').hide();
        });
    }

    // Slider change handler - sync with input and auto-apply scale
    $('#slider-scale-factor').on('input', function() {
        const scale = parseFloat($(this).val());
        // Update input field
        $('#input-scale-factor').val(scale.toFixed(2));
        // Update display value
        $('#slider-scale-value').text(scale.toFixed(2));
        
        // Debounce auto-apply to avoid too many requests
        if (scaleSliderDebounceTimer) {
            clearTimeout(scaleSliderDebounceTimer);
        }
        scaleSliderDebounceTimer = setTimeout(function() {
            applyScale(scale, false); // Don't update slider (already synced)
        }, 300); // Wait 300ms after user stops dragging
    });

    // Input change handler - sync with slider
    $('#input-scale-factor').on('input change', function() {
        const val = $(this).val();
        if (!val) return;
        const scale = parseFloat(val);
        if (isNaN(scale) || scale <= 0) return;
        
        // Clamp to slider range
        const clampedScale = Math.max(0.1, Math.min(5, scale));
        // Update slider
        $('#slider-scale-factor').val(clampedScale);
        $('#slider-scale-value').text(clampedScale.toFixed(2));
    });

    // Apply button handler
    $('#btn-apply-scale').click(function() {
        const val = $('#input-scale-factor').val();
        if (!val) {
            $('#scale-status').text('Please enter a scale > 0');
            return;
        }
        const scale = parseFloat(val);
        applyScale(scale, true);
    });



    // Magnify View Logic - REMOVED (Integrated into Focus View)
    // function showMagnifyView(cx, cy, cz, traces) { ... }

    $('#close-magnify-modal').click(function() {
        $('#magnify-modal').hide();
    });

    function togglePlay() {
        if (isPlaying) {
            // Stop playback
            isPlaying = false;
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
            if (playAnimationFrameId !== null) {
                cancelAnimationFrame(playAnimationFrameId);
                playAnimationFrameId = null;
            }
            $('#play-pause').text('▶ Play').removeClass('playing');
            // When playback stops, loadFrame will handle UI updates
            // Full 3D view updates happen only when user interacts with 3D view
        } else {
            // Start playback
            // Ensure we're not at the end
            if (currentFrame >= totalFrames - 1) {
                currentFrame = 0;
            }

            isPlaying = true;
            $('#play-pause').text('⏸ Pause').addClass('playing');

            // Update current frame first to ensure display is correct
            loadFrame(currentFrame);

            // Use simple setInterval for playback, similar to test_video
            // This avoids the overhead of requestAnimationFrame for frame-based playback
            const frameDelay = 1000 / fps; // Match test_video: use exact fps timing

            playInterval = setInterval(() => {
                if (!isPlaying) return;

                if (currentFrame < totalFrames - 1) {
                    currentFrame++;
                    loadFrame(currentFrame);
                } else {
                    // Reached end, stop playback
                    togglePlay();
                }
            }, frameDelay);
        }
    }

    function loadMesh() {
        $.get('/api/mesh', function(data) {
            meshData = data;
            renderMesh();
        });
    }

    function renderMesh() {
        meshTrace = {
            type: 'mesh3d',
            x: meshData.x,
            y: meshData.y,
            z: meshData.z,
            i: meshData.i,
            j: meshData.j,
            k: meshData.k,
            color: 'lightgray',
            opacity: 0.8,
            flatshading: true,
            hoverinfo: 'none',
            name: 'Mesh'
        };

        scatterTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: [],
            y: [],
            z: [],
            marker: { size: 8, color: 'red' },
            name: 'Object Points',
            hoverinfo: 'text',
            text: []
        };

        humanTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: [],
            y: [],
            z: [],
            marker: { size: 10, color: 'lime' },
            name: 'Human Joints',
            hoverinfo: 'text',
            text: []
        };

        layout = {
            scene: {
                aspectmode: 'data',
                dragmode: 'orbit'
            },
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true,
            legend: { x: 0, y: 1 }
        };

        Plotly.newPlot('3d-viewer', [meshTrace, scatterTrace, humanTrace], layout, {responsive: true});

        document.getElementById('3d-viewer').on('plotly_click', function(data) {
            const point = data.points[0];
            
            if (currentMode === 'view') return;

            // Select Mode: Add new or Select existing
            if (currentMode === 'select') {
                // Helper function to check if ALL selected points have at least one annotation
                // Returns true if all points have 2D tracking OR human keypoint (or both)
                function allPointsHaveAnnotation() {
                    if (selectedPoints.length === 0) return true; // No points selected, allow selection
                    
                    // Check each selected point
                    for (const pt of selectedPoints) {
                        const objIdx = pt.index;
                        // Check if this point has 2D point selected (in pending2DPoints) or completed tracking
                        const has2DPointSelected = !!pending2DPoints[objIdx];
                        const has2DTrackCompleted = objPointToTrack[objIdx] && 
                                                    Object.keys(objPointToTrack[objIdx]).length > 0;
                        const has2DTrack = has2DPointSelected || has2DTrackCompleted;
                        // Check if this point has human keypoint mapping
                        const hasHumanKp = !!objPointToJoint[objIdx];
                        
                        // If this point has neither 2D point selected/tracked nor human keypoint, return false
                        if (!has2DTrack && !hasHumanKp) {
                            return false;
                        }
                    }
                    return true; // All points have at least one annotation
                }

                // Click on Mesh (curveNumber 0) -> Add new point
                if (point.curveNumber === 0) {
                    const idx = point.pointNumber;
                    const x = meshData.x[idx];
                    const y = meshData.y[idx];
                    const z = meshData.z[idx];
                    
                    // Check if already selected
                    const existingIdx = selectedPoints.findIndex(p => p.index === idx);
                    if (existingIdx === -1) {
                        // This is a NEW point - check if all previously selected points have annotation
                        if (!allPointsHaveAnnotation()) {
                            alert("Please select 2D tracking or human keypoints first.");
                            return;
                        }
                        // Add new point
                        selectedPoints.push({index: idx, x: x, y: y, z: z});
                        activeObjectPointIndex = idx; // Set as active
                    } else {
                        // Just select existing point - no check needed
                        activeObjectPointIndex = idx;
                    }
                    
                    // Reset pending 2D point when switching active object point
                    pending2DPoint = null;
                    
                    // Defer UI update to prevent freezing the browser event loop
                    setTimeout(updateSelection, 10);
                    
                    // NOTE: intentionally do NOT call updateFrame() here
                    // to avoid triggering 2D redraw on every 3D click.
                }
                
                // Click on Object Point (curveNumber 1) -> Select it
                else if (point.curveNumber === 1) {
                    const idx = point.pointNumber; // Index in selectedPoints array
                    if (idx >= 0 && idx < selectedPoints.length) {
                        activeObjectPointIndex = selectedPoints[idx].index;
                        pending2DPoint = null;
                        setTimeout(updateSelection, 10);
                        // updateFrame();
                    }
                }
            }

            // Delete Mode: Click on Selected Points (curveNumber 1)
            else if (currentMode === 'delete') {
                if (point.curveNumber === 1) { // Object Point
                    const idx = point.pointNumber; // Index in selectedPoints array
                    if (idx >= 0 && idx < selectedPoints.length) {
                        const objIdx = selectedPoints[idx].index;
                        
                        // Check if this point has 2D point selected (in pending2DPoints) or completed tracking
                        const has2DPointSelected = !!pending2DPoints[objIdx];
                        const has2DTrackCompleted = objPointToTrack[objIdx] && 
                                                    Object.keys(objPointToTrack[objIdx]).length > 0;
                        const has2DTrack = has2DPointSelected || has2DTrackCompleted;
                        const hasHumanKp = !!objPointToJoint[objIdx];
                        
                        // If the point has neither 2D point selected/tracked nor human keypoint,
                        // completely remove it from selectedPoints
                        if (!has2DTrack && !hasHumanKp) {
                            // Remove from selectedPoints array
                            selectedPoints.splice(idx, 1);
                            
                            // Clear any pending 2D points for this object
                            if (pending2DPoints[objIdx]) {
                                delete pending2DPoints[objIdx];
                            }
                            
                            // Clear any keyframes for this object
                            if (jointKeyframesByObj[objIdx]) {
                                delete jointKeyframesByObj[objIdx];
                            }
                            if (visibilityKeyframesByObj[objIdx]) {
                                delete visibilityKeyframesByObj[objIdx];
                            }
                            
                            if (activeObjectPointIndex === objIdx) {
                                activeObjectPointIndex = -1;
                                pending2DPoint = null;
                            }
                            
                            setTimeout(updateSelection, 10);
                            return;
                        }
                        
                        // If the point has 2D tracking or human keypoint,
                        // keep the existing logic: hide from current frame onward
                        addJointKeyframe(objIdx, currentFrame, null);
                        addVisibilityKeyframe(objIdx, currentFrame, false);
                        applyJointMappingForCurrentFrame();

                        // For any 2D tracks linked to this object point,
                        // only clear tracking results from the current
                        // frame onward, preserving earlier frames.
                        clearTrackFromFrame(objIdx, currentFrame);

                        if (activeObjectPointIndex === objIdx) {
                            activeObjectPointIndex = -1;
                            pending2DPoint = null;
                        }

                        setTimeout(updateSelection, 10);
                        // Likewise, avoid updateFrame() on delete to isolate 3D performance
                    }
                }
            }
        });
        
        document.getElementById('3d-viewer').oncontextmenu = function() { return false; };
    }

    function updateSelection() {
        // Prepare data for Object Points (Trace 1), respecting
        // per-frame visibility so that points deleted at a
        // given frame disappear from that frame onward.
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        const text = [];

        selectedPoints.forEach(p => {
            if (!isObjectVisibleAtFrame(p.index, currentFrame)) {
                return; // hidden from this frame onward
            }

            x.push(p.x);
            y.push(p.y);
            z.push(p.z);

            colors.push(p.index === activeObjectPointIndex ? 'blue' : 'red');

            let status = '';
            if (objPointToJoint[p.index]) status = ` (Linked to ${objPointToJoint[p.index]})`;
            else if (objPointToTrack[p.index]) status = ` (Tracked)`;
            text.push(`ID: ${p.index}${status}`);
        });

        // Prepare data for Human Keypoints (Trace 2)
        const linkedObjIndices = Object.keys(objPointToJoint)
            .map(Number)
            .filter(idx => isObjectVisibleAtFrame(idx, currentFrame));
        
        const hx = [];
        const hy = [];
        const hz = [];
        const ht = [];
        
        linkedObjIndices.forEach(idx => {
            const pt = selectedPoints.find(p => p.index === idx);
            if (pt) {
                hx.push(pt.x);
                hy.push(pt.y);
                hz.push(pt.z);
                ht.push(`Linked to: ${objPointToJoint[idx]}`);
            }
        });

        // Update traces directly on the graph div to avoid recreating the plot
        const gd = document.getElementById('3d-viewer');
        if (!gd || !meshTrace) return;

        // Update local trace objects
        scatterTrace.x = x;
        scatterTrace.y = y;
        scatterTrace.z = z;
        scatterTrace.text = text;
        scatterTrace.marker.color = colors;

        humanTrace.x = hx;
        humanTrace.y = hy;
        humanTrace.z = hz;
        humanTrace.text = ht;

        // Use Plotly.react for efficient update that handles data changes correctly
        // Pass the same meshTrace reference to avoid re-processing the mesh
        Plotly.react(gd, [meshTrace, scatterTrace, humanTrace], gd.layout);
    }
    
    function updateProgressDisplay(frameNum = currentFrame) {
        // Only update UI elements for responsive display during dragging
        $('#frame-display').text('Frame: ' + frameNum);
        if (!isDragging) {
            // Only update slider if not dragging to avoid conflict
            $('#frame-slider').val(frameNum);
        }
    }

        // 加载指定帧（用于播放和拖动滑条）
        function loadFrame(frameNum) {
            currentFrame = Math.max(0, Math.min(frameNum, totalFrames - 1));

            const videoFrame = $('#video-frame')[0];
            const modalVideoFrame = $('#modal-video-frame')[0];
            const frameSrc = '/api/frame/' + currentFrame;

            // 主视频区域：直接切换图片，不做任何变暗/过渡效果，避免闪烁
            if (videoFrame) {
                videoFrame.src = frameSrc;
            }

            // 如果标注弹窗打开，则同步更新 2D 视图的图片
            if (modalVideoFrame && $('#annotation-modal').is(':visible')) {
                modalVideoFrame.src = frameSrc;
                // 每次图片加载完成后重绘 2D 叠加（追踪点等）
                modalVideoFrame.onload = function() {
                    update2DOverlay();
                    this.onload = null;
                };
            }

            $('#frame-display').text('Frame: ' + currentFrame);
            if (!isDragging) {
                $('#frame-slider').val(currentFrame);
            }

            // 预加载相邻帧
            preloadFrames(currentFrame);

            // 每次切换帧时，同时更新 3D/2D 状态：
            // 1) 依据关键帧重建当前帧的人体关节映射
            // 2) 重新绘制左侧 3D 物体点（包含按帧可见性）
            // 3) 更新主视频和 2D 叠加，让追踪点随帧变化
            applyJointMappingForCurrentFrame();
            updateSelection();
            updateMainOverlay();
            update2DOverlay();
        }

        // 帧加载错误处理
        $('#video-frame').on('error', function() {
            console.error('帧加载失败:', currentFrame);
        });

        // 不再在 load 时修改透明度，避免闪烁
        $('#video-frame').on('load', function() {
            // no-op
        });

    function updateFrame() {
        // Update frame display immediately for responsiveness
        $('#frame-display').text('Frame: ' + currentFrame);
        if (!isDragging) {
            // Only update slider if not dragging to avoid conflict
            $('#frame-slider').val(currentFrame);
        }
        
        // 使用稳定 URL，交给浏览器和后端缓存处理
        const frameSrc = '/api/frame/' + currentFrame;
        
        // Update video frame images when用户跳帧/暂停时查看
        const videoFrame = $('#video-frame')[0];
        const modalVideoFrame = $('#modal-video-frame')[0];
        
        // Update main video frame（不再做变暗效果）
        if (videoFrame) {
            const currentBaseSrc = videoFrame.src.split('?')[0];
            const newBaseSrc = frameSrc;
            
            // Update src if frame number changed or force reload
            if (currentBaseSrc !== newBaseSrc || !videoFrame.complete) {
                // Update src
                videoFrame.src = frameSrc;
                
                // Handle load completion
                videoFrame.onload = function() {
                    this.onload = null; // Clean up
                };
                
                // Handle load error
                videoFrame.onerror = function() {
                    this.onerror = null; // Clean up
                };
            } else {
                // Force reload even if src looks the same
                videoFrame.src = frameSrc;
            }
        }
        
        // Update modal video frame (only if modal is open)
        if (modalVideoFrame && $('#annotation-modal').is(':visible')) {
            modalVideoFrame.src = frameSrc;
        }

        // 更新主视频和 2D 视图上的叠加（追踪点、pending 点）
        updateMainOverlay();
        update2DOverlay();
    }

    // 在主视频上绘制追踪点
    function updateMainOverlay() {
        const canvas = document.getElementById('main-video-overlay');
        const img = document.getElementById('video-frame');
        if (!canvas || !img) return;

        // 确保图像已经加载好再绘制
        if (!img.complete || img.naturalWidth === 0) {
            img.onload = function() {
                updateMainOverlay();
                img.onload = null;
            };
            return;
        }

        // 调整 canvas 尺寸匹配当前显示尺寸
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;

        let renderWidth, renderHeight, offsetX, offsetY;
        if (clientRatio > naturalRatio) {
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }

        // 在主视频上画所有追踪点（与 modal 一致，当前激活点高亮）
        for (const [objIdx, tracks] of Object.entries(objPointToTrack)) {
            const pt = tracks[currentFrame] || tracks[String(currentFrame)];
            if (!pt) continue;

            const [x, y] = pt;
            const displayX = x * (renderWidth / img.naturalWidth) + offsetX;
            const displayY = y * (renderHeight / img.naturalHeight) + offsetY;

            ctx.beginPath();
            ctx.arc(displayX, displayY, 4, 0, 2 * Math.PI);

            if (parseInt(objIdx) === activeObjectPointIndex) {
                ctx.fillStyle = '#00ff00';
                ctx.lineWidth = 2;
            } else {
                ctx.fillStyle = '#008800';
                ctx.lineWidth = 1;
            }

            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.stroke();
        }
    }

    // 仅负责根据当前帧在 modal 里的 canvas 上画追踪点
    function update2DOverlay() {
        const canvas = document.getElementById('modal-video-overlay');
        if (!canvas || !$('#annotation-modal').is(':visible')) {
            // Modal not open, skip canvas update
            return;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Pre-calculate image geometry to avoid layout thrashing in loop
        const img = document.getElementById('modal-video-frame');
        if (!img || img.naturalWidth === 0) return; 
        
        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;
        
        let renderWidth, renderHeight, offsetX, offsetY;
        
        if (clientRatio > naturalRatio) {
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }

        // Draw tracked points for ALL objects
        for (const [objIdx, tracks] of Object.entries(objPointToTrack)) {
            const pt = tracks[currentFrame] || tracks[String(currentFrame)];
            if (pt) {
                 const [x, y] = pt;
                 
                 const displayX = x * (renderWidth / img.naturalWidth) + offsetX;
                 const displayY = y * (renderHeight / img.naturalHeight) + offsetY;
                 
                 ctx.beginPath();
                 ctx.arc(displayX, displayY, 5, 0, 2 * Math.PI);
                 
                 // Highlight if active
                 if (parseInt(objIdx) === activeObjectPointIndex) {
                     ctx.fillStyle = '#00ff00'; // Lime green for active
                     ctx.lineWidth = 3;
                 } else {
                     ctx.fillStyle = '#008800'; // Darker green for others
                     ctx.lineWidth = 1;
                 }
                 
                 ctx.fill();
                 ctx.strokeStyle = 'white';
                 ctx.stroke();
                 
                 // Label
                 ctx.fillStyle = 'white';
                 ctx.font = '12px Arial';
                 ctx.fillText(`ID: ${objIdx}`, displayX + 8, displayY + 4);
            }
        }
        
        // Draw all pending 2D points (red), one per object index
        for (const [objIdx, p] of Object.entries(pending2DPoints)) {
             let dx = p.displayX;
             let dy = p.displayY;

             if (dx === undefined) {
                 dx = p.x * (renderWidth / img.naturalWidth) + offsetX;
                 dy = p.y * (renderHeight / img.naturalHeight) + offsetY;
             }

             ctx.beginPath();
             ctx.arc(dx, dy, 5, 0, 2 * Math.PI);
             ctx.fillStyle = 'red';
             ctx.fill();
             ctx.strokeStyle = 'white';
             ctx.lineWidth = 2;
             ctx.stroke();
             ctx.fillStyle = 'white';
             ctx.font = '12px Arial';
             ctx.fillText(`Pending ${objIdx}`, dx + 8, dy + 4);
        }
    }

    function saveMergedAnnotations(callback) {
        // Prepare payload with all in-memory annotations
        const payload = {
            joint_keyframes: jointKeyframesByObj,
            visibility_keyframes: visibilityKeyframesByObj,
            tracks: objPointToTrack,
            total_frames: totalFrames,
            last_frame: currentFrame // Optional: limit saving to current frame
        };

        $.ajax({
            url: '/api/save_merged_annotations',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(payload),
            success: function(response) {
                if (response.status === 'success') {
                    console.log('Merged annotations saved to:', response.path);
                    if (callback) callback(true);
                } else {
                    console.error('Failed to save merged annotations:', response);
                    if (callback) callback(false);
                }
            },
            error: function(xhr, status, error) {
                console.error('Error saving merged annotations:', error);
                if (callback) callback(false);
            }
        });
    }

    function saveAnnotation() {
        // Construct human_keypoints: { jointName: {index, x, y, z} }
        const humanKeypointsExport = {};
        for (const [objIdx, jointName] of Object.entries(objPointToJoint)) {
            const pt = selectedPoints.find(p => p.index === parseInt(objIdx));
            if (pt) {
                humanKeypointsExport[jointName] = {
                    index: pt.index,
                    x: pt.x, y: pt.y, z: pt.z
                };
            }
        }

        $.ajax({
            url: '/api/save_annotation',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                frame: currentFrame,
                object_points: selectedPoints,
                human_keypoints: humanKeypointsExport,
                tracks: objPointToTrack
            }),
            success: function(response) {
                alert('Saved annotation for frame ' + currentFrame);
            },
            error: function(xhr) {
                alert('Save failed: ' + (xhr.responseJSON?.error || 'Unknown error'));
            }
        });
    }

    function saveAllAnnotations() {
        const isStatic = $('#static-object').is(':checked');

        $.ajax({
            url: '/api/save_merged_annotations',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                is_static_object: isStatic,
                total_frames: totalFrames,
                // Save only up to the frame the user is
                // currently on when clicking "Save All".
                last_frame: currentFrame,
                // Per-object 3D joint keyframes over time
                joint_keyframes: jointKeyframesByObj,
                // Per-object visibility keyframes over time
                visibility_keyframes: visibilityKeyframesByObj,
                // 2D tracks for each object point
                tracks: objPointToTrack
            }),
            success: function(response) {
                const outPath = response.path || 'kp_record_merged.json';
                alert('Saved merged annotations to:\n' + outPath);
            },
            error: function(xhr) {
                alert('Save-all failed: ' + (xhr.responseJSON?.error || 'Unknown error'));
            }
        });
    }
    
    // Management UI
    $('#btn-manage').click(function() {
        renderAnnotationList();
        $('#management-panel').show();
    });
    
    $('#close-manager').click(function() {
        $('#management-panel').hide();
    });
    
    function renderAnnotationList() {
        const tbody = $('#annotation-table tbody');
        tbody.empty();
        
        selectedPoints.forEach(pt => {
            const idx = pt.index;
            let type = 'None';
            let target = '-';
            
            if (objPointToJoint[idx]) {
                type = 'Human Joint';
                target = objPointToJoint[idx];
            } else if (objPointToTrack[idx]) {
                type = '2D Track';
                const track = objPointToTrack[idx];
                const currentPos = track[currentFrame] || track[String(currentFrame)];
                target = currentPos ? `Frame ${currentFrame}: (${Math.round(currentPos[0])}, ${Math.round(currentPos[1])})` : 'No track this frame';
            }
            
            const tr = $('<tr>');
            tr.append($('<td>').text(idx));
            tr.append($('<td>').text(type));
            tr.append($('<td>').text(target));

            const actions = $('<td>');

            if (type === 'Human Joint') {
                const editJointBtn = $('<button>').text('Edit Joint').click(function() {
                    activeObjectPointIndex = idx;
                    if (!$('#annotation-modal').is(':visible')) {
                        $('#annotation-modal').show();
                    }
                    $('#tab-human').click();
                    $('#selected-joint-display').text(objPointToJoint[idx] || 'None');
                    updateSelection();
                });
                actions.append(editJointBtn);
            } else if (type === '2D Track') {
                const edit2DBtn = $('<button>').text('Edit 2D').click(function() {
                    activeObjectPointIndex = idx;
                    if (!$('#annotation-modal').is(':visible')) {
                        $('#annotation-modal').show();
                    }
                    $('#tab-2d').click();

                    const track = objPointToTrack[idx] || {};
                    const currentPos = track[currentFrame] || track[String(currentFrame)];
                    if (currentPos) {
                        pending2DPoint = {
                            x: currentPos[0],
                            y: currentPos[1]
                        };
                        $('#2d-status').text(
                            `Current: (${Math.round(currentPos[0])}, ${Math.round(currentPos[1])}). ` +
                            'Click new position on the image, then press "Track 2D Point" to retrack from this frame.'
                        );
                    } else {
                        pending2DPoint = null;
                        $('#2d-status').text(
                            'No 2D point on this frame. Click on the image to choose one, then press "Track 2D Point".'
                        );
                    }

                    $('#btn-track-2d').prop('disabled', false);
                    update2DOverlay();
                });
                actions.append(edit2DBtn);
            }

            const deleteBtn = $('<button>').text('Delete').click(function() {
                // From the management panel, deletion of a Human Joint
                // should only affect from the current frame onward, not
                // earlier frames. We therefore add a null keyframe for
                // this object and hide the 3D point from this frame
                // onward, instead of removing the underlying 3D point
                // globally.
                if (type === 'Human Joint') {
                    addJointKeyframe(idx, currentFrame, null);
                    addVisibilityKeyframe(idx, currentFrame, false);
                    applyJointMappingForCurrentFrame();
                }

                // For any 2D tracks, only clear tracking results from
                // the current frame onward, keeping earlier frames intact.
                clearTrackFromFrame(idx, currentFrame);

                if (activeObjectPointIndex === idx) activeObjectPointIndex = -1;
                updateSelection();
                updateFrame();
                renderAnnotationList();
            });
            actions.append(deleteBtn);

            tr.append(actions);

            tbody.append(tr);
        });
    }

    function togglePoint(idx, x, y, z) {
        // Deprecated in favor of explicit add/delete modes
    }

    let focusMode = 'view'; // 'view', 'magnify'
    let focusTraces = []; // Store traces for magnification

    // Focus Hand Button Handler
    $('#btn-focus-hand').click(function() {
        $('#focus-frame-idx').text(currentFrame);
        $('#focus-modal').show();
        // Reset mode
        focusMode = 'view';
        $('#btn-focus-magnify').css('background-color', '#17a2b8').text('Magnify');
        
        updateFocusView(currentFrame);
    });
    
    // Focus Magnify Button Handler
    $('#btn-focus-magnify').click(function() {
        if (focusMode === 'view') {
            focusMode = 'magnify';
            $(this).css('background-color', '#ffc107').text('Click to Magnify');
        } else {
            focusMode = 'view';
            $(this).css('background-color', '#17a2b8').text('Magnify');
        }
    });
    
    // Focus Reset Button Handler
    $('#btn-focus-reset').click(function() {
        const gd = document.getElementById('focus-viewer');
        // Reset camera to default
        Plotly.relayout(gd, {
            'scene.camera.center': {x: 0, y: 0, z: 0},
            'scene.camera.eye': {x: 1.25, y: 1.25, z: 1.25},
            'scene.camera.up': {x: 0, y: 0, z: 1}
        });
    });

    $('#close-focus-modal').click(function() {
        $('#focus-modal').hide();
    });

    function updateFocusView(frameIdx) {
        const gd = document.getElementById('focus-viewer');
        // Plotly.purge(gd); // Optional: Clear previous

        $.get('/api/focus_hand/' + frameIdx, function(data) {
            const human = data.human;
            const object = data.object;
            const camera = data.camera;

            const humanTrace = {
                type: 'mesh3d',
                x: human.x, y: human.y, z: human.z,
                i: human.i, j: human.j, k: human.k,
                color: 'pink', opacity: 1.0,
                name: 'Human'
            };

            const objectTrace = {
                type: 'mesh3d',
                x: object.x, y: object.y, z: object.z,
                i: object.i, j: object.j, k: object.k,
                color: 'lightblue', opacity: 0.8,
                name: 'Object'
            };
            
            focusTraces = [humanTrace, objectTrace];

            const layout = {
                scene: {
                    aspectmode: 'data',
                    camera: camera
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                showlegend: true
            };

            Plotly.newPlot('focus-viewer', focusTraces, layout, { responsive: true });
            
            // Attach click handler for magnification
            // Remove previous handlers to avoid duplicates
            gd.removeAllListeners('plotly_click');
            
            gd.on('plotly_click', function(data) {
                if (focusMode === 'magnify') {
                    try {
                        const point = data.points[0];
                        const x = point.x;
                        const y = point.y;
                        const z = point.z;
                        
                        let minX = Infinity, maxX = -Infinity;
                        let minY = Infinity, maxY = -Infinity;
                        let minZ = Infinity, maxZ = -Infinity;
                        
                        [human, object].forEach(mesh => {
                            if (mesh.x && mesh.x.length > 0) {
                                for(let i=0; i<mesh.x.length; i++) {
                                    const v = mesh.x[i]; if(v < minX) minX = v; if(v > maxX) maxX = v;
                                }
                                for(let i=0; i<mesh.y.length; i++) {
                                    const v = mesh.y[i]; if(v < minY) minY = v; if(v > maxY) maxY = v;
                                }
                                for(let i=0; i<mesh.z.length; i++) {
                                    const v = mesh.z[i]; if(v < minZ) minZ = v; if(v > maxZ) maxZ = v;
                                }
                            }
                        });
                        
                        if (!isFinite(minX) || !isFinite(maxX)) {
                            console.error("Invalid mesh bounds");
                            return;
                        }
                        
                        const centerX = (minX + maxX) / 2;
                        const centerY = (minY + maxY) / 2;
                        const centerZ = (minZ + maxZ) / 2;
                        
                        const sizeX = maxX - minX;
                        const sizeY = maxY - minY;
                        const sizeZ = maxZ - minZ;
                        const maxDim = Math.max(sizeX, sizeY, sizeZ) || 1.0; // Avoid div by zero
                        
                        const normX = (x - centerX) / maxDim;
                        const normY = (y - centerY) / maxDim;
                        const normZ = (z - centerZ) / maxDim;
                        
                        // Use _fullLayout to get the actual current camera state
                        const scene = gd._fullLayout ? gd._fullLayout.scene : null;
                        const currentEye = scene ? scene.camera.eye : {x: 1.25, y: 1.25, z: 1.25};
                        
                        const ex = currentEye.x;
                        const ey = currentEye.y;
                        const ez = currentEye.z;
                        const len = Math.sqrt(ex*ex + ey*ey + ez*ez) || 1.0;
                        
                        // Zoom distance: 0.4 units away from the target point
                        const zoomDist = 0.4; 
                        
                        const newEyeX = normX + (ex / len) * zoomDist;
                        const newEyeY = normY + (ey / len) * zoomDist;
                        const newEyeZ = normZ + (ez / len) * zoomDist;
                        
                        Plotly.relayout(gd, {
                            'scene.camera.center': {x: normX, y: normY, z: normZ},
                            'scene.camera.eye': {x: newEyeX, y: newEyeY, z: newEyeZ}
                        });
                        
                        // Reset button state
                        focusMode = 'view';
                        $('#btn-focus-magnify').css('background-color', '#17a2b8').text('Magnify');
                    } catch (e) {
                        console.error("Error in magnify click handler:", e);
                    }
                }
            });

        }).fail(function(xhr) {
            alert('Error loading focus view: ' + (xhr.responseJSON?.error || 'Unknown error'));
            $('#focus-modal').hide();
        });
    }
});
