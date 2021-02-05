mod application;
mod display;

use anyhow::Result;
use application::Application;
use flexi_logger::DeferredNow;
use flexi_logger::Logger;
use flexi_logger::Record;
use std::fmt::Write as FmtWrite;
use textwrap::{termwidth, Options};

fn main() -> Result<()> {
    let result = run();
    if let Err(ref error) = result {
        log::error!(
            "Application exited unsuccessfully!\n{:?}\n\nroot cause: {:?}",
            error,
            error.root_cause()
        );
    }
    result
}

fn run() -> Result<()> {
    Logger::with_env_or_str("info")
        .format(multiline_format)
        .start()?;

    let app = Application::initialize()?;
    app.main_loop()
}

/// A formatting function for lines which automaticaly wrap on the terminal
/// width.
fn multiline_format(
    w: &mut dyn std::io::Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    let size = termwidth().min(74);
    let wrap_options = Options::new(size)
        .initial_indent("┏ ")
        .subsequent_indent("┃ ");

    let mut full_line = String::new();
    writeln!(
        full_line,
        "{} [{}] [{}:{}]",
        record.level(),
        now.now().format("%H:%M:%S%.6f"),
        record.file().unwrap_or("<unnamed>"),
        record.line().unwrap_or(0),
    )
    .expect("unable to format first log line");

    write!(&mut full_line, "{}", &record.args())
        .expect("unable to format log!");

    writeln!(w, "{}", textwrap::fill(&full_line, wrap_options))
}
