"""Optional AppKit toolbar; the subtitle editor remains in its WKWebView.

All Cocoa work runs on the AppKit main thread. Native actions dispatch a fixed
command vocabulary; document paths and executable JavaScript never cross back
from toolbar state updates.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Any

log = logging.getLogger(__name__)
_controller = None


def normalize_state(value: object) -> dict:
    """Keep untrusted bridge input small, typed and presentation-only."""
    data = value if isinstance(value, dict) else {}
    return {
        "title": str(data.get("title") or "SubForge")[:180],
        "status": str(data.get("status") or "字幕工作室")[:90],
        "appearance": data.get("appearance")
        if data.get("appearance") in ("light", "dark")
        else "system",
        "can_export": data.get("can_export") is True,
        "can_inspect": data.get("can_inspect") is True,
        "inspector_open": data.get("inspector_open") is True,
        "running": data.get("running") is True,
        "sidebar_collapsed": data.get("sidebar_collapsed") is True,
        "navigation": data.get("navigation")
        if data.get("navigation")
        in ("import", "transcribe", "subtitle", "free-models", "llm-logs", "settings")
        else "import",
    }


def get_desktop_state() -> dict:
    controller = _controller
    return dict(controller.capabilities) if controller is not None else {"toolbar": False}


def sync_desktop_state(value: object) -> dict:
    controller = _controller
    if controller is None:
        return {"toolbar": False}
    from PyObjCTools import AppHelper

    AppHelper.callAfter(controller.applyState, normalize_state(value))
    return get_desktop_state()


def _sidebar_brand(AppKit: Any):
    """A compact native identity block, with text hidden in the icon rail."""
    brand = AppKit.NSStackView.alloc().initWithFrame_(AppKit.NSZeroRect)
    brand.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
    brand.setAlignment_(AppKit.NSLayoutAttributeCenterY)
    brand.setSpacing_(10)
    mark = AppKit.NSImageView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 30, 30))
    mark.setImage_(
        AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "captions.bubble.fill", None
        )
    )
    mark.setContentTintColor_(AppKit.NSColor.controlAccentColor())
    mark.setSymbolConfiguration_(
        AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
            24, AppKit.NSFontWeightMedium
        )
    )
    mark.widthAnchor().constraintEqualToConstant_(30).setActive_(True)
    mark.heightAnchor().constraintEqualToConstant_(30).setActive_(True)
    brand.addArrangedSubview_(mark)
    brand_text = AppKit.NSStackView.alloc().initWithFrame_(AppKit.NSZeroRect)
    brand_text.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
    brand_text.setAlignment_(AppKit.NSLayoutAttributeLeading)
    brand_text.setSpacing_(3)
    for text, size, color in (
        ("SubForge", 14, AppKit.NSColor.labelColor()),
        ("字幕工作室", 11, AppKit.NSColor.secondaryLabelColor()),
    ):
        label = AppKit.NSTextField.labelWithString_(text)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(size, AppKit.NSFontWeightMedium))
        label.setTextColor_(color)
        brand_text.addArrangedSubview_(label)
    brand.addArrangedSubview_(brand_text)
    return brand, brand_text


def _window_material(AppKit: Any, content_class: Any):
    """One continuous glass substrate, with a semantic older-system fallback."""
    glass = getattr(AppKit, "NSGlassEffectView", None)
    if glass is not None:
        view = glass.alloc().initWithFrame_(AppKit.NSZeroRect)
        content = content_class.alloc().initWithFrame_(AppKit.NSZeroRect)
        content.setWantsLayer_(True)
        view.setContentView_(content)
        view.setCornerRadius_(0)
        view.setStyle_(AppKit.NSGlassEffectViewStyleRegular)
        return view, content, True
    backdrop = AppKit.NSVisualEffectView.alloc().initWithFrame_(AppKit.NSZeroRect)
    backdrop.setMaterial_(AppKit.NSVisualEffectMaterialSidebar)
    backdrop.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
    backdrop.setState_(AppKit.NSVisualEffectStateFollowsWindowActiveState)
    content = content_class.alloc().initWithFrame_(AppKit.NSZeroRect)
    backdrop.addSubview_(content)
    _pin_edges(AppKit, content, backdrop)
    return backdrop, content, False


def _tone_window_material(AppKit: Any, content: Any):
    dark = content.effectiveAppearance().bestMatchFromAppearancesWithNames_(
        [AppKit.NSAppearanceNameAqua, AppKit.NSAppearanceNameDarkAqua]
    ) == AppKit.NSAppearanceNameDarkAqua
    rgb = (27, 31, 37) if dark else (246, 247, 249)
    content.setWantsLayer_(True)
    content.layer().setBackgroundColor_(
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            *(channel / 255 for channel in rgb), 0.32
        ).CGColor()
    )


def _sync_toolbar_appearance(buttons: dict, appearance: Any):
    for button in buttons.values():
        button.superview().setAppearance_(appearance)
        button.setAppearance_(appearance)


def _pin_edges(AppKit: Any, child: Any, parent: Any):
    child.setTranslatesAutoresizingMaskIntoConstraints_(False)
    AppKit.NSLayoutConstraint.activateConstraints_([
        child.leadingAnchor().constraintEqualToAnchor_(parent.leadingAnchor()),
        child.trailingAnchor().constraintEqualToAnchor_(parent.trailingAnchor()),
        child.topAnchor().constraintEqualToAnchor_(parent.topAnchor()),
        child.bottomAnchor().constraintEqualToAnchor_(parent.bottomAnchor()),
    ])


def _workspace_panel(AppKit: Any, webview: Any):
    panel = AppKit.NSView.alloc().initWithFrame_(AppKit.NSZeroRect)
    panel.setWantsLayer_(True)
    panel.layer().setCornerRadius_(18)
    panel.layer().setShadowColor_(AppKit.NSColor.blackColor().CGColor())
    panel.layer().setShadowOpacity_(0.10)
    panel.layer().setShadowRadius_(12)
    panel.layer().setShadowOffset_(AppKit.NSMakeSize(0, -3))
    clip = AppKit.NSView.alloc().initWithFrame_(AppKit.NSZeroRect)
    clip.setWantsLayer_(True)
    clip.layer().setCornerRadius_(18)
    clip.layer().setCornerCurve_("continuous")
    clip.layer().setMasksToBounds_(True)
    panel.addSubview_(clip)
    clip.addSubview_(webview)
    _pin_edges(AppKit, clip, panel)
    _pin_edges(AppKit, webview, clip)
    return panel


def _layout_navigation(AppKit: Any, navigation: dict, collapsed: bool):
    inset = 0 if collapsed else 8
    for button, selection, _ in navigation.values():
        bounds = selection.bounds()
        button.setFrame_(AppKit.NSMakeRect(
            inset, 0, max(0, bounds.size.width - 2 * inset), bounds.size.height
        ))


def _navigation_classes(AppKit: Any, objc: Any):
    """Native buttons keep semantics; their parent paints the sole state surface."""
    import time

    class SubForgeNavigationButton(AppKit.NSButton):
        def updateTrackingAreas(self):
            objc.super(SubForgeNavigationButton, self).updateTrackingAreas()
            for area in self.trackingAreas():
                self.removeTrackingArea_(area)
            self.addTrackingArea_(AppKit.NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                AppKit.NSZeroRect,
                AppKit.NSTrackingMouseEnteredAndExited | AppKit.NSTrackingActiveInKeyWindow
                | AppKit.NSTrackingInVisibleRect,
                self, None,
            ))

        def mouseEntered_(self, event):
            self.surface.hover(self.tag())

        def mouseExited_(self, event):
            self.surface.hover(None)

    class SubForgeNavigationSurface(AppKit.NSView):
        @objc.python_method
        def configure(self, controller):
            self.controller = controller
            self.selected = None
            self.hovered = None
            self.timer = None
            self.motion = None
            self.fade_time = 0
            self.hover_values = {}
            self.hover_start = {}
            self.hover_time = 0

        @objc.python_method
        def hover(self, tag):
            self.sampleHover()
            self.hover_start = dict(self.hover_values)
            self.hovered = tag
            self.hover_time = time.monotonic()
            self.startTimer()

        @objc.python_method
        def sampleHover(self):
            fraction = min(1, (time.monotonic() - self.hover_time) / 0.12)
            for button, _, _ in self.controller.navigation.values():
                tag = button.tag()
                initial = self.hover_start.get(tag, 0)
                self.hover_values[tag] = initial + (float(tag == self.hovered) - initial) * fraction
            return fraction < 1

        @objc.python_method
        def targetRect(self):
            if self.selected not in self.controller.navigation:
                return None
            _, row, _ = self.controller.navigation[self.selected]
            return row.convertRect_toView_(row.bounds(), self)

        @objc.python_method
        def currentRect(self):
            target = self.targetRect()
            if self.motion is None or target is None:
                return target
            start, started = self.motion
            progress = min(1, (time.monotonic() - started) / 0.20)
            eased = 1 - (1 - progress) ** 3
            rect = AppKit.NSMakeRect(*[
                a + (b - a) * eased for a, b in zip(
                    (start.origin.x, start.origin.y, start.size.width, start.size.height),
                    (target.origin.x, target.origin.y, target.size.width, target.size.height),
                    strict=True,
                )
            ])
            if progress >= 1:
                self.motion = None
            return rect

        @objc.python_method
        def select(self, key):
            previous = self.currentRect()
            changed = self.selected != key
            self.selected = key
            if self.controller.capabilities.get("reduce_motion"):
                self.motion = None
                if changed:
                    self.fade_time = time.monotonic()
            elif changed and previous is not None:
                # Retarget from the visible position; there is never an animation queue.
                self.motion = (previous, time.monotonic())
            self.setNeedsDisplay_(True)
            if self.motion is not None or time.monotonic() - self.fade_time < 0.08:
                self.startTimer()

        @objc.python_method
        def startTimer(self):
            if self.timer is None:
                self.timer = AppKit.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                    1 / 60, self, "tick:", None, True
                )
                AppKit.NSRunLoop.mainRunLoop().addTimer_forMode_(
                    self.timer, AppKit.NSRunLoopCommonModes
                )

        def tick_(self, timer):
            hovering = self.sampleHover()
            self.currentRect()
            self.setNeedsDisplay_(True)
            if self.motion is None and not hovering and time.monotonic() - self.fade_time >= 0.08:
                timer.invalidate()
                self.timer = None

        def layout(self):
            objc.super(SubForgeNavigationSurface, self).layout()
            self.setNeedsDisplay_(True)

        def drawRect_(self, dirty):
            if not hasattr(self, "controller"):
                return
            self.sampleHover()
            contrast = self.controller.capabilities.get("increase_contrast")
            for key, (button, row, _) in self.controller.navigation.items():
                if key == self.selected:
                    continue
                alpha = self.hover_values.get(button.tag(), 0) * (0.09 if contrast else 0.045)
                AppKit.NSColor.labelColor().colorWithAlphaComponent_(alpha).setFill()
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    row.convertRect_toView_(row.bounds(), self), 10, 10
                ).fill()
            rect = self.currentRect()
            if rect is not None:
                button = self.controller.navigation[self.selected][0]
                alpha = (0.16 if contrast else 0.10) + 0.025 * self.hover_values.get(button.tag(), 0)
                alpha *= min(1, (time.monotonic() - self.fade_time) / 0.08)
                AppKit.NSColor.controlAccentColor().colorWithAlphaComponent_(alpha).setFill()
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 10, 10).fill()

    return SubForgeNavigationButton, SubForgeNavigationSurface


def install_native_toolbar(window) -> bool:
    """Install after the webview is ready, retaining the original window owner."""
    if _controller is not None:
        return bool(get_desktop_state().get("toolbar"))
    if sys.platform != "darwin":
        return False
    import AppKit as _AppKit
    import objc as _objc
    from PyObjCTools import AppHelper

    # PyObjC resolves AppKit symbols lazily from runtime metadata.
    AppKit: Any = _AppKit
    objc: Any = _objc

    NavigationButton, NavigationSurface = _navigation_classes(AppKit, objc)
    ready = threading.Event()

    class SubForgeWindowContent(AppKit.NSView):
        def viewDidChangeEffectiveAppearance(self):
            objc.super(SubForgeWindowContent, self).viewDidChangeEffectiveAppearance()
            _tone_window_material(AppKit, self)

    class SubForgeToolbarController(
        AppKit.NSObject, protocols=[objc.protocolNamed("NSToolbarDelegate")]
    ):
        @objc.python_method
        def setup(self):
            self.window = window.native
            self.webview = self.window.contentView()
            self.original_style_mask = self.window.styleMask()
            self.buttons = {}
            self.items = {}
            self.capabilities = {"toolbar": False, "liquid_glass": False}
            self.navigation = {}
            self.sidebar_labels = []
            self.readAccessibility()
            self.installSidebar()
            toolbar = AppKit.NSToolbar.alloc().initWithIdentifier_("SubForge.Workspace")
            toolbar.setDelegate_(self)
            toolbar.setAllowsUserCustomization_(False)
            toolbar.setAutosavesConfiguration_(False)
            toolbar.setDisplayMode_(AppKit.NSToolbarDisplayModeIconOnly)
            self.toolbar = toolbar
            self.window.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)
            self.window.setTitlebarAppearsTransparent_(True)
            self.window.setTitlebarSeparatorStyle_(AppKit.NSTitlebarSeparatorStyleNone)
            self.window.setToolbar_(toolbar)
            self.capabilities["toolbar"] = True
            self.applyState(normalize_state({}))
            AppKit.NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
                self,
                "accessibilityChanged:",
                AppKit.NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification,
                None,
            )
            log.info("Native toolbar installed: %s", self.capabilities)
            log.info(
                "Native button metrics: %s",
                {
                    key: {
                        "bezel": button.bezelStyle(),
                        "bordered": bool(button.isBordered()),
                        "size": str(button.frame().size),
                    }
                    for key, button in self.buttons.items()
                },
            )

        @objc.python_method
        def installSidebar(self):
            self.material, self.content, glass = _window_material(AppKit, SubForgeWindowContent)
            self.material.setFrame_(self.webview.frame())
            self.window.setStyleMask_(
                self.original_style_mask | AppKit.NSWindowStyleMaskFullSizeContentView
            )
            self.window.setContentView_(self.material)
            self.capabilities["sidebar_glass"] = glass
            self.sidebar_host = NavigationSurface.alloc().initWithFrame_(AppKit.NSZeroRect)
            self.sidebar_host.configure(self)
            self.sidebar_content = self.sidebar_host
            self.content.addSubview_(self.sidebar_host)
            self.workspace_panel = _workspace_panel(AppKit, self.webview)
            self.content.addSubview_(self.workspace_panel)
            self.sidebar_host.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.workspace_panel.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.sidebar_width = self.sidebar_host.widthAnchor().constraintEqualToConstant_(204)
            layout = self.window.contentLayoutGuide()
            AppKit.NSLayoutConstraint.activateConstraints_([
                self.sidebar_host.leadingAnchor().constraintEqualToAnchor_constant_(
                    self.content.leadingAnchor(), 8
                ),
                self.sidebar_host.topAnchor().constraintEqualToAnchor_constant_(
                    layout.topAnchor(), 8
                ),
                self.sidebar_host.bottomAnchor().constraintEqualToAnchor_constant_(
                    self.content.bottomAnchor(), -12
                ),
                self.sidebar_width,
                self.workspace_panel.leadingAnchor().constraintEqualToAnchor_constant_(
                    self.sidebar_host.trailingAnchor(), 8
                ),
                self.workspace_panel.trailingAnchor().constraintEqualToAnchor_constant_(
                    self.content.trailingAnchor(), -12
                ),
                self.workspace_panel.topAnchor().constraintEqualToAnchor_constant_(
                    layout.topAnchor(), 8
                ),
                self.workspace_panel.bottomAnchor().constraintEqualToAnchor_constant_(
                    self.content.bottomAnchor(), -12
                ),
            ])
            stack = AppKit.NSStackView.alloc().initWithFrame_(AppKit.NSZeroRect)
            stack.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
            stack.setAlignment_(AppKit.NSLayoutAttributeLeading)
            stack.setSpacing_(5)
            stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
            self.sidebar_content.addSubview_(stack)
            AppKit.NSLayoutConstraint.activateConstraints_(
                [
                    stack.leadingAnchor().constraintEqualToAnchor_constant_(
                        self.sidebar_content.leadingAnchor(), 12
                    ),
                    stack.trailingAnchor().constraintEqualToAnchor_constant_(
                        self.sidebar_content.trailingAnchor(), -12
                    ),
                    stack.topAnchor().constraintEqualToAnchor_constant_(
                        self.sidebar_content.topAnchor(), 18
                    ),
                    stack.bottomAnchor().constraintEqualToAnchor_constant_(
                        self.sidebar_content.bottomAnchor(), -16
                    ),
                ]
            )

            def caption(text):
                label = AppKit.NSTextField.labelWithString_(text)
                label.setFont_(
                    AppKit.NSFont.systemFontOfSize_weight_(11, AppKit.NSFontWeightSemibold)
                )
                label.setTextColor_(AppKit.NSColor.secondaryLabelColor())
                stack.addArrangedSubview_(label)
                self.sidebar_labels.append(label)
                stack.setCustomSpacing_afterView_(10, label)

            brand, brand_text = _sidebar_brand(AppKit)
            self.sidebar_labels.append(brand_text)
            stack.addArrangedSubview_(brand)
            stack.setCustomSpacing_afterView_(26, brand)
            caption("工作区")
            specs = [
                (10, "import", "导入素材", "folder", "1"),
                (11, "transcribe", "语音转录", "waveform", "2"),
                (12, "subtitle", "字幕工作区", "captions.bubble", "3"),
                (13, "free-models", "免费模型", "square.grid.2x2", ""),
                (14, "llm-logs", "诊断日志", "waveform.path.ecg", ""),
                (15, "settings", "设置", "gearshape", ","),
            ]
            for tag, key, title, symbol, shortcut in specs:
                if tag == 13:
                    caption("资源")
                if tag == 14:
                    spacer = AppKit.NSView.alloc().initWithFrame_(AppKit.NSZeroRect)
                    spacer.setContentHuggingPriority_forOrientation_(
                        1, AppKit.NSLayoutConstraintOrientationVertical
                    )
                    stack.addArrangedSubview_(spacer)
                host = AppKit.NSView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 188, 36))
                button = NavigationButton.buttonWithTitle_image_target_action_(
                    title,
                    AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                        symbol, title
                    ),
                    self,
                    "navigate:",
                )
                button.setTag_(tag)
                button.surface = self.sidebar_host
                button.setBordered_(False)
                button.cell().setHighlightsBy_(AppKit.NSNoCellMask)
                button.setFocusRingType_(AppKit.NSFocusRingTypeExterior)
                button.setAlignment_(AppKit.NSTextAlignmentLeft)
                button.setFont_(
                    AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightMedium)
                )
                button.setImageHugsTitle_(True)
                button.setSymbolConfiguration_(
                    AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                        17, AppKit.NSFontWeightRegular
                    )
                )
                button.setFrame_(AppKit.NSMakeRect(8, 0, 172, 36))
                button.setAutoresizingMask_(AppKit.NSViewWidthSizable)
                button.setAccessibilityLabel_(title)
                button.setToolTip_(title + (f"（⌘{shortcut}）" if shortcut else ""))
                if shortcut:
                    button.setKeyEquivalent_(shortcut)
                    button.setKeyEquivalentModifierMask_(AppKit.NSEventModifierFlagCommand)
                host.addSubview_(button)
                stack.addArrangedSubview_(host)
                host.widthAnchor().constraintEqualToAnchor_(stack.widthAnchor()).setActive_(True)
                host.heightAnchor().constraintEqualToConstant_(36).setActive_(True)
                self.navigation[key] = (button, host, title)
                if tag == 12:
                    stack.setCustomSpacing_afterView_(24, host)
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
            self.capabilities["native_sidebar"] = True

        def navigate_(self, sender):
            key = {
                10: "import",
                11: "transcribe",
                12: "subtitle",
                13: "free-models",
                14: "llm-logs",
                15: "settings",
            }.get(sender.tag())
            if key:
                self.window.makeFirstResponder_(self.webview)
                self.sendEvent("subforge:command", "navigate:" + key)

        @objc.python_method
        def readAccessibility(self):
            workspace = AppKit.NSWorkspace.sharedWorkspace()
            self.capabilities.update(
                {
                    "reduce_transparency": bool(
                        workspace.accessibilityDisplayShouldReduceTransparency()
                    ),
                    "reduce_motion": bool(workspace.accessibilityDisplayShouldReduceMotion()),
                    "increase_contrast": bool(
                        workspace.accessibilityDisplayShouldIncreaseContrast()
                    ),
                }
            )

        def accessibilityChanged_(self, notification):
            self.readAccessibility()
            self.sidebar_host.select(self.sidebar_host.selected)
            self.sendEvent("subforge:desktop", self.capabilities)

        @objc.python_method
        def sendEvent(self, event, detail):
            # Use WKWebView directly here. pywebview.evaluate_js waits for a main-
            # thread callback, which would deadlock an AppKit button action.
            script = f"window.dispatchEvent(new CustomEvent({json.dumps(event)},{{detail:{json.dumps(detail)}}}));"
            self.webview.evaluateJavaScript_completionHandler_(script, None)

        def command_(self, sender):
            command = {1: "sidebar", 2: "import", 3: "inspector", 4: "export", 5: "cancel"}.get(
                sender.tag()
            )
            if command:
                self.window.makeFirstResponder_(self.webview)
                self.sendEvent("subforge:command", command)

        @objc.python_method
        def button(self, command, label, symbol):
            image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                symbol, label
            )
            button = AppKit.NSButton.buttonWithTitle_image_target_action_(
                label, image, self, "command:"
            )
            button.setTag_(command)
            button.setControlSize_(AppKit.NSControlSizeLarge)
            button.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightMedium))
            button.setImageHugsTitle_(True)
            button.setSymbolConfiguration_(
                AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                    14, AppKit.NSFontWeightRegular
                )
            )
            button.setButtonType_(
                AppKit.NSButtonTypePushOnPushOff
                if command == 3
                else AppKit.NSButtonTypeMomentaryPushIn
            )
            button.setBordered_(True)
            modern = button.respondsToSelector_("setTintProminence:")
            # Use the native glass button bezel: it owns padding, hit feedback,
            # focus and disabled treatments. No nested glass backplate or frames.
            button.setBezelStyle_(
                AppKit.NSBezelStyleGlass if modern else AppKit.NSBezelStyleTexturedRounded
            )
            if modern:
                button.setPrefersCompactControlSizeMetrics_(False)
                button.setTintProminence_(AppKit.NSTintProminenceAutomatic)
                self.capabilities["liquid_glass"] = True
            button.setToolTip_(
                {
                    1: "显示或隐藏侧边栏",
                    2: "导入视频、音频或字幕（⌘O）",
                    3: "显示或隐藏处理选项",
                    4: "选择格式并导出字幕",
                    5: "取消当前任务",
                }[command]
            )
            button.setAccessibilityLabel_(label or button.toolTip())
            button.sizeToFit()
            # Keep native intrinsic metrics and enough optical padding for
            # localized titles, including builds running on older macOS.
            text_size = AppKit.NSString.stringWithString_(label).sizeWithAttributes_(
                {AppKit.NSFontAttributeName: button.font()}
            )
            width = max(button.frame().size.width, text_size.width + (56 if label else 40))
            height = max(button.frame().size.height, 36)
            button.setFrameSize_(AppKit.NSMakeSize(width, height))
            self.buttons[command] = button
            return button

        def toolbarDefaultItemIdentifiers_(self, toolbar):
            return [
                "sidebar",
                AppKit.NSToolbarFlexibleSpaceItemIdentifier,
                "import",
                "processing",
                AppKit.NSToolbarSpaceItemIdentifier,
                "cancel",
                "export",
            ]

        def toolbarAllowedItemIdentifiers_(self, toolbar):
            return self.toolbarDefaultItemIdentifiers_(toolbar)

        def toolbar_itemForItemIdentifier_willBeInsertedIntoToolbar_(
            self, toolbar, identifier, inserted
        ):
            specs = {
                "sidebar": (1, "", "sidebar.left"),
                "import": (2, "导入", "square.and.arrow.down"),
                "processing": (3, "处理选项", "slider.horizontal.3"),
                "cancel": (5, "取消任务", "stop.circle"),
                "export": (4, "导出", "square.and.arrow.up"),
            }
            if identifier not in specs:
                return None
            command, label, symbol = specs[identifier]
            item = AppKit.NSToolbarItem.alloc().initWithItemIdentifier_(identifier)
            item.setLabel_(label or "侧边栏")
            item.setBordered_(False)
            button = self.button(command, label, symbol)
            # A layout-only host keeps the explicit native button bezel and
            # intrinsic metrics intact, without adding a second glass surface.
            host = AppKit.NSView.alloc().initWithFrame_(button.frame())
            host.addSubview_(button)
            button.setTranslatesAutoresizingMaskIntoConstraints_(False)
            AppKit.NSLayoutConstraint.activateConstraints_(
                [
                    button.leadingAnchor().constraintEqualToAnchor_(host.leadingAnchor()),
                    button.trailingAnchor().constraintEqualToAnchor_(host.trailingAnchor()),
                    button.topAnchor().constraintEqualToAnchor_(host.topAnchor()),
                    button.bottomAnchor().constraintEqualToAnchor_(host.bottomAnchor()),
                    button.widthAnchor().constraintGreaterThanOrEqualToConstant_(
                        button.frame().size.width
                    ),
                    button.heightAnchor().constraintGreaterThanOrEqualToConstant_(
                        button.frame().size.height
                    ),
                ]
            )
            item.setView_(host)
            item.setTag_(command)
            item.setTarget_(self)
            item.setAction_("command:")
            item.setAutovalidates_(False)
            if identifier == "sidebar":
                item.setNavigational_(True)
            self.items[identifier] = item
            return item

        @objc.python_method
        def setCancelVisible(self, visible):
            item = self.items.get("cancel")
            if item is not None and item.respondsToSelector_("setHidden:"):
                item.setHidden_(not visible)
                return
            identifiers = [item.itemIdentifier() for item in self.toolbar.items()]
            if visible and "cancel" not in identifiers:
                self.toolbar.insertItemWithItemIdentifier_atIndex_(
                    "cancel", identifiers.index("export")
                )
            elif not visible and "cancel" in identifiers:
                self.toolbar.removeItemAtIndex_(identifiers.index("cancel"))

        @objc.python_method
        def applyState(self, data):
            self.window.setTitle_(data["title"])
            self.window.setSubtitle_(data["status"])
            appearance = {
                "light": AppKit.NSAppearanceNameAqua,
                "dark": AppKit.NSAppearanceNameDarkAqua,
            }.get(data["appearance"])
            self.window.setAppearance_(
                AppKit.NSAppearance.appearanceNamed_(appearance) if appearance else None
            )
            self.material.setAppearance_(self.window.appearance())
            self.content.setAppearance_(self.window.appearance())
            self.sidebar_content.setAppearance_(self.window.appearance())
            _tone_window_material(AppKit, self.content)
            _sync_toolbar_appearance(self.buttons, self.window.appearance())
            collapsed = data["sidebar_collapsed"]
            width = 64 if collapsed else 204
            changing_width = self.sidebar_width.constant() != width
            AppKit.NSAnimationContext.beginGrouping()
            context = AppKit.NSAnimationContext.currentContext()
            context.setDuration_(
                0.2 if changing_width and not self.capabilities.get("reduce_motion") else 0
            )
            context.setAllowsImplicitAnimation_(True)
            self.sidebar_width.setConstant_(width)
            for label in self.sidebar_labels:
                label.setHidden_(collapsed)
            for key, (button, _row, title) in self.navigation.items():
                selected = key == data["navigation"]
                button.setContentTintColor_(
                    AppKit.NSColor.controlAccentColor() if selected else AppKit.NSColor.labelColor()
                )
                button.setTitle_("" if collapsed else title)
                button.setImagePosition_(AppKit.NSImageOnly if collapsed else AppKit.NSImageLeft)
                button.setAlignment_(
                    AppKit.NSTextAlignmentCenter if collapsed else AppKit.NSTextAlignmentLeft
                )
                button.setAccessibilityValue_("已选中" if selected else "")
            self.content.layoutSubtreeIfNeeded()
            self.content.setNeedsDisplay_(True)
            _layout_navigation(AppKit, self.navigation, collapsed)
            AppKit.NSAnimationContext.endGrouping()
            self.sidebar_host.select(data["navigation"])
            for key, enabled in (
                (4, data["can_export"]),
                (3, data["can_inspect"]),
                (5, data["running"]),
            ):
                if key in self.buttons:
                    self.buttons[key].setEnabled_(enabled)
                    identifier = {3: "processing", 4: "export", 5: "cancel"}[key]
                    self.items[identifier].setEnabled_(enabled)
            self.setCancelVisible(data["running"])
            if 3 in self.buttons:
                inspector = self.buttons[3]
                selected = data["inspector_open"] and data["can_inspect"]
                inspector.setState_(
                    AppKit.NSControlStateValueOn if selected else AppKit.NSControlStateValueOff
                )
                inspector.setToolTip_("收起处理选项" if selected else "展开处理选项")
                if inspector.respondsToSelector_("setTintProminence:"):
                    inspector.setTintProminence_(AppKit.NSTintProminenceNone)
                    inspector.setBezelColor_(AppKit.NSColor.controlColor() if selected else None)
            if 4 in self.buttons:
                export = self.buttons[4]
                if export.respondsToSelector_("setTintProminence:"):
                    export.setBezelColor_(None)
                    export.setTintProminence_(
                        AppKit.NSTintProminencePrimary
                        if data["can_export"]
                        else AppKit.NSTintProminenceAutomatic
                    )
                    foreground = (
                        AppKit.NSColor.whiteColor()
                        if data["can_export"]
                        else AppKit.NSColor.labelColor()
                    )
                    export.setAttributedTitle_(
                        AppKit.NSAttributedString.alloc().initWithString_attributes_(
                            "导出",
                            {
                                AppKit.NSFontAttributeName: export.font(),
                                AppKit.NSForegroundColorAttributeName: foreground,
                            },
                        )
                    )
                    metrics = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                        14, AppKit.NSFontWeightRegular
                    )
                    colors = AppKit.NSImageSymbolConfiguration.configurationWithPaletteColors_(
                        [foreground]
                    )
                    export.setSymbolConfiguration_(
                        metrics.configurationByApplyingConfiguration_(colors)
                    )
                else:
                    export.setBezelColor_(
                        AppKit.NSColor.controlAccentColor() if data["can_export"] else None
                    )

    def install():
        global _controller
        controller = SubForgeToolbarController.alloc().init()
        try:
            controller.setup()
            _controller = controller
        except Exception:
            log.exception("Native toolbar unavailable; retaining the web toolbar")
            if getattr(window, "native", None):
                window.native.setToolbar_(None)
                if getattr(controller, "webview", None):
                    controller.webview.setTranslatesAutoresizingMaskIntoConstraints_(True)
                    window.native.setStyleMask_(controller.original_style_mask)
                    window.native.setContentView_(controller.webview)
                    window.native.setOpaque_(True)
                    window.native.setBackgroundColor_(AppKit.NSColor.windowBackgroundColor())
        finally:
            ready.set()

    AppHelper.callAfter(install)
    ready.wait(5)
    return bool(get_desktop_state().get("toolbar"))
