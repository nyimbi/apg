from typing import Dict, Optional, List
from iso8583.message import ISO8583Message
import logging
from dataclasses import dataclass

@dataclass
class RoutingRule:
    """Contains routing configuration for a specific BIN range"""
    bin_prefix: str
    destination: str
    priority: int = 0
    active: bool = True

class RoutingEngine:
    """
    Handles routing logic for ISO8583 messages based on configured BIN ranges.

    Attributes:
        routing_table (Dict[str, RoutingRule]): Mapping of BIN prefixes to routing rules
        logger (logging.Logger): Logger instance for routing events
    """

    def __init__(self, routing_table: Dict[str, str]):
        """
        Initialize routing engine with routing table configuration.

        Args:
            routing_table: Dictionary mapping BIN prefixes to destinations
        """
        self.logger = logging.getLogger(__name__)
        self.routing_table: Dict[str, RoutingRule] = {
            prefix: RoutingRule(prefix, dest)
            for prefix, dest in routing_table.items()
        }

    def get_destination(self, message: ISO8583Message) -> str:
        """
        Determine routing destination based on message PAN.

        Args:
            message: ISO8583 message containing PAN in field 2

        Returns:
            str: Destination identifier for routing

        Raises:
            ValueError: If no valid routing found for message PAN
            KeyError: If required message field is missing
        """
        try:
            pan = str(message.fields[2])
            bin_range = pan[:6]

            # Check exact matches first
            if bin_range in self.routing_table:
                rule = self.routing_table[bin_range]
                if rule.active:
                    self.logger.info(f"Found exact routing match for BIN {bin_range}")
                    return rule.destination

            # Check prefix matches
            matching_rules = []
            for rule in self.routing_table.values():
                if (rule.active and
                    bin_range.startswith(rule.bin_prefix)):
                    matching_rules.append(rule)

            if matching_rules:
                # Sort by priority and prefix length
                rule = sorted(matching_rules,
                            key=lambda x: (-x.priority, -len(x.bin_prefix)))[0]
                self.logger.info(f"Found prefix routing match for BIN {bin_range}")
                return rule.destination

            self.logger.error(f"No routing found for BIN {bin_range}")
            raise ValueError(f"No routing found for BIN {bin_range}")

        except KeyError:
            self.logger.error("Missing PAN field in message")
            raise KeyError("Message missing required PAN field (2)")

    def add_rule(self, bin_prefix: str, destination: str,
                 priority: int = 0) -> None:
        """Add a new routing rule"""
        self.routing_table[bin_prefix] = RoutingRule(
            bin_prefix, destination, priority)

    def disable_rule(self, bin_prefix: str) -> None:
        """Disable an existing routing rule"""
        if bin_prefix in self.routing_table:
            self.routing_table[bin_prefix].active = False

    def get_active_rules(self) -> List[RoutingRule]:
        """Return list of all active routing rules"""
        return [rule for rule in self.routing_table.values()
                if rule.active]
