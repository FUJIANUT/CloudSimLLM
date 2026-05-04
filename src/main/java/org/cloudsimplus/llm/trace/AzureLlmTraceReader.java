package org.cloudsimplus.llm.trace;

import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.workload.LlmModelSpec;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for the Azure LLM Inference trace (Patel et al., ASPLOS'24,
 * <a href="https://github.com/Azure/AzurePublicDataset">Azure public dataset</a>).
 *
 * <p>Expected CSV columns (one row per request):
 * <pre>timestamp,context_tokens,generated_tokens[,model][,tenant]</pre>
 * </p>
 *
 * <p>Each row materializes one {@link LlmCloudlet}. The trace timestamps are
 * preserved into {@link LlmCloudlet#onArrival(double)} so the broker can submit
 * cloudlets at the right simulation time.</p>
 */
public class AzureLlmTraceReader {

    private final LlmModelSpec defaultModel;
    private LlmCloudlet.SloClass defaultClass = LlmCloudlet.SloClass.INTERACTIVE;
    private long firstTimestampNanos = -1;

    public AzureLlmTraceReader(LlmModelSpec defaultModel) {
        this.defaultModel = defaultModel;
    }

    public AzureLlmTraceReader setDefaultClass(LlmCloudlet.SloClass c) { this.defaultClass = c; return this; }

    public List<TraceEntry> read(Path csv) throws Exception {
        List<TraceEntry> out = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(csv)) {
            String header = br.readLine();             // skip header row
            if (header == null) return out;
            String line;
            long idCounter = 0;
            while ((line = br.readLine()) != null) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] f = line.split(",");
                if (f.length < 3) continue;
                long tsNs = parseTimestamp(f[0]);
                if (firstTimestampNanos < 0) firstTimestampNanos = tsNs;
                int sIn = Integer.parseInt(f[1].trim());
                int sOut = Integer.parseInt(f[2].trim());
                if (sIn <= 0 || sOut <= 0) continue;
                double simSec = (tsNs - firstTimestampNanos) / 1e9;
                LlmCloudlet c = new LlmCloudlet(idCounter++, defaultModel, sIn, sOut, defaultClass);
                out.add(new TraceEntry(simSec, c));
            }
        }
        return out;
    }

    private long parseTimestamp(String s) {
        s = s.trim();
        // Heuristic: integer → nanoseconds; ISO → parse to nanos.
        if (s.matches("^\\d+$")) {
            long v = Long.parseLong(s);
            return v < 1_000_000_000_000L ? v * 1_000_000_000L : v;        // sec → ns if too small
        }
        return java.time.Instant.parse(s).getEpochSecond() * 1_000_000_000L;
    }

    public record TraceEntry(double submitAtSec, LlmCloudlet cloudlet) {}
}
