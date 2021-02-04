use anyhow::{Context, Result};
use log;
use std::sync::Arc;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{
    layers_list, ApplicationInfo, Instance, InstanceExtensions, Version,
};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub fn create_instance() -> Result<Arc<Instance>> {
    if ENABLE_VALIDATION_LAYERS && !check_debug_layers()? {
        log::warn!("requested validation layers are unavailable")
    }

    let supported_extensions = InstanceExtensions::supported_by_core()
        .context("unable to get supported instance extensions")?;
    let required_extensions = required_extensions();
    log::info!("supported extensions: {:?}", supported_extensions);
    log::info!("required extensions: {:?}", required_extensions);

    let app_info = ApplicationInfo {
        application_name: Some("Vulkan Experiments".into()),
        application_version: Some(Version {
            major: 1,
            minor: 0,
            patch: 0,
        }),
        engine_name: Some("no engine".into()),
        engine_version: None,
    };

    Ok(Instance::new(Some(&app_info), &required_extensions, None)?)
}

fn check_debug_layers() -> Result<bool> {
    let available_layers: Vec<String> = layers_list()?
        .map(|layer| layer.name().to_owned())
        .collect();

    log::info!("available debug layers {:?}", available_layers);

    let all_available = VALIDATION_LAYERS.iter().all(|required_layer| {
        available_layers.contains(&required_layer.to_string())
    });
    Ok(all_available)
}

fn required_extensions() -> InstanceExtensions {
    let mut required_extensions = vulkano_win::required_extensions();
    if ENABLE_VALIDATION_LAYERS {
        required_extensions.ext_debug_utils = true;
    }
    required_extensions
}

pub fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let severity = MessageSeverity {
        error: true,
        warning: true,
        information: true,
        verbose: true,
    };

    let msgtype = MessageType {
        general: true,
        performance: true,
        validation: true,
    };

    DebugCallback::new(instance, severity, msgtype, |msg| match msg.severity {
        MessageSeverity { error: true, .. } => {
            log::error!("Vulkan Debug Callback\n{:?}", msg.description)
        }
        MessageSeverity { warning: true, .. } => {
            log::warn!("Vulkan Debug Callback\n{:?}", msg.description)
        }
        MessageSeverity {
            information: true, ..
        } => {
            log::info!("Vulkan Debug Callback\n{:?}", msg.description);
        }
        MessageSeverity { verbose: true, .. } => {
            log::debug!("Vulkan Debug Callback\n{:?}", msg.description);
        }
        _ => {
            log::debug!("Vulkan Debug Callback\n{:?}", msg.description)
        }
    })
    .ok()
}
